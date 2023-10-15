import functools

import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from ..ops import (voxelization, topk_nms)
from ..util import cuda_cast, force_fp32, rle_encode, get_iou, dice_loss, HungarianMatcher
from .blocks import MLP, ResidualBlock, UBlock, UBlock_tiney, query_instance


class ICFMNet(nn.Module):

    def __init__(self,
                 channels=32,
                 num_blocks=7,
                 semantic_only=False,
                 semantic_classes=20,
                 instance_classes=18,
                 sem2ins_classes=[],
                 ignore_label=-100,
                 candidate_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 query_cfg=None,
                 fixed_modules=[],
                 use_transformer=False,
                 transformer_before=False,
                 stop_trans=2,
                 ):
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks
        self.semantic_only = semantic_only
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.sem2ins_classes = sem2ins_classes
        self.ignore_label = ignore_label
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.query_cfg = query_cfg
        self.fixed_modules = fixed_modules

        self.candidate_nms_N = candidate_cfg.nms_N
        self.candidate_nms_R = candidate_cfg.nms_R
        self.candidate_nms_thres = candidate_cfg.nms_thres
        self.candidate_nms_localThres = candidate_cfg.nms_localThres
        self.candidate_nms_maxActive_f = candidate_cfg.nms_maxActive_f

        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_channels = [channels * (i + 1) for i in range(num_blocks)]

        # backbone
        self.input_conv = spconv.SparseSequential(spconv.SubMConv3d(6, channels, kernel_size=3, padding=1, bias=False, indice_key='subm1'))
        self.unet = UBlock(block_channels, norm_fn, 2, block, indice_key_id=1, use_transformer=use_transformer, transformer_before=transformer_before, nsample=32, stop_trans=stop_trans)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())

        # point-wise prediction
        self.semantic_linear = MLP(channels, semantic_classes, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=2)

        self.catFeat_linear = MLP(channels + 3, channels, norm_fn=norm_fn, num_layers=2)
        self.candidate_linear = MLP(channels, 1, norm_fn=norm_fn, num_layers=3)
        self.query_catFeat_linear = MLP(channels + 3, channels, norm_fn=norm_fn, num_layers=2)
        self.mergeCandidate_linear = MLP(channels + 3, 1, norm_fn=norm_fn, num_layers=2)

        # self.tiny_unet_inputlayer = MLP(channels, 4 * channels, norm_fn=norm_fn, num_layers=1)
        self.tiny_unet_inputlayer = spconv.SparseSequential(spconv.SubMConv3d(channels, 4 * channels, kernel_size=3, padding=1, bias=False, indice_key='subm1'))
        self.tiny_unet = UBlock_tiney([4 * channels, 4 * channels, 5 * channels, 6 * channels, 8 * channels], norm_fn, 2, block, indice_key_id=13)
        self.tiny_unet_outputlayer = spconv.SparseSequential(norm_fn(4 * channels), nn.ReLU())

        # query
        if not semantic_only:
            self.query_loss_weight = torch.tensor(query_cfg.query_loss_weight)

            query_cfg.in_channel = 4 * channels
            self.query_instance_layer = query_instance(**query_cfg)

            self.matcher = HungarianMatcher(cost_weight=query_cfg.cost_weight)

            self.layer_num = query_cfg.num_decode + 1

            # if not query_cfg.use_feat_query:
            #     self.query = nn.Embedding(candidate_cfg.nms_N, query_cfg.d_model)

        self.init_weights()

        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False

        # 遍历模型参数，检查 requires_grad 是否为 False
        # for name, param in self.named_parameters():
        #     if not param.requires_grad:
        #         print(f'{name} has requires_grad = False')
        #     else:
        #         print(f'{name} has requires_grad = True')

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()

    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
        return self

    def forward(self, batch, return_loss=False):
        if return_loss:
            return self.forward_train(**batch)
        else:
            return self.forward_test(**batch)

    @cuda_cast
    def forward_train(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                      semantic_labels, instance_labels, instance_pointnum, instance_cls, instances3D_labels,
                      pt_offset_labels, spatial_shape, batch_size, **kwargs):
        losses = {}

        output_feats, sparse_batch_idxs, sparse_id, sparse_coords, _ = self.forward_backbone(
            torch.cat((feats, coords_float), 1), p2v_map, voxel_coords, spatial_shape, batch_size
        )
        candidate_scores, semantic_scores, pt_offsets, merge_pred, (candidate_beforeMerge_id, candidate_afterMerge_id), output_feats, query_feats = self.forward_point_wise(
            output_feats, v2p_map, sparse_batch_idxs, sparse_coords, sparse_id
        )

        # point wise losses
        point_wise_loss = self.point_wise_loss(
            candidate_scores, semantic_scores, semantic_labels, pt_offsets, instance_labels, pt_offset_labels, merge_pred, candidate_beforeMerge_id, batch_idxs
        )
        losses.update(point_wise_loss)

        # query instance
        if not self.semantic_only:
            # candidate_afterMerge_id = candidate_beforeMerge_id
            candidate_batch_idxs = batch_idxs[candidate_afterMerge_id]

            # if not self.query_cfg.use_feat_query:
            #     query_feats = self.query.weight.unsqueeze(0).repeat(B, 1, 1)
            batch_inst_loss = {}
            for bs in range(batch_size):
                prediction = self.query_instance_layer(
                    output_feats[bs], query_feats[bs],
                    v2p_map.long(), sparse_batch_idxs, bs)

                # query instance losses
                pred_labels = prediction['labels']
                pred_scores = prediction['scores']
                pred_masks = prediction['masks']

                inst = self.get_instance3D(semantic_labels, instance_labels, candidate_afterMerge_id[candidate_batch_idxs == bs], torch.where(batch_idxs == bs)[0])

                query_inst_loss = self.query_instance_layer_loss(pred_labels, pred_scores, pred_masks, inst)

                for loss_name, loss_value in query_inst_loss.items():
                    if loss_name in batch_inst_loss.keys():
                        batch_inst_loss[loss_name] += loss_value
                    else:
                        batch_inst_loss[loss_name] = loss_value

            batch_inst_loss = {loss_name: (loss_value / batch_size) for loss_name, loss_value in batch_inst_loss.items()}

            losses.update(batch_inst_loss)

        return self.parse_losses(losses)

    def point_wise_loss(self, candidate_scores, semantic_scores, semantic_labels, pt_offsets, instance_labels, pt_offset_labels, merge_pred, candidate_id, batch_idxs):
        losses = {}

        ''' '''
        self.invaild_fore = 0
        for fm in self.foreground_mask:
            fil = instance_labels[fm]
            self.invaild_fore += torch.unique(fil, return_counts=True)[1].max() / fil.shape[0]
        self.invaild_fore /= len(self.foreground_mask)
        ''' '''

        semantic_loss = F.cross_entropy(semantic_scores, semantic_labels, ignore_index=self.ignore_label)
        losses['semantic_loss'] = semantic_loss

        pos_inds = instance_labels != instance_labels.new_full((1,), self.ignore_label)

        if pos_inds.sum() == 0:
            offset_loss = 0 * pt_offsets.sum()
            candidate_loss = 0 * pt_offsets.sum()
        else:
            '''offset loss'''
            offset_loss = F.l1_loss(pt_offsets[pos_inds], pt_offset_labels[pos_inds, 0:3], reduction='sum') / pos_inds.sum()

            '''candidate loss'''
            candidate_loss = F.l1_loss(candidate_scores[pos_inds], pt_offset_labels[pos_inds, 3], reduction='sum') / pos_inds.sum()

        losses['offset_loss'] = offset_loss
        losses['sCandidate_loss'] = candidate_loss

        candidate_instance_labels = instance_labels[candidate_id]
        candidate_batch_idxs = batch_idxs[candidate_id]
        if merge_pred.shape[0] == 0 or candidate_instance_labels.shape[0] == 0:
            candidate_merge_loss = 0 * pt_offsets.sum()
        else:
            vaild_inds = (candidate_instance_labels != self.ignore_label)
            merge_pred = merge_pred[vaild_inds, :][:, vaild_inds]
            candidate_instance_labels = candidate_instance_labels[vaild_inds]
            candidate_batch_idxs = candidate_batch_idxs[vaild_inds]

            merge_gt = torch.zeros_like(merge_pred)
            candidate_instance_labels_max = candidate_instance_labels.max() + 1
            for i in range(candidate_instance_labels_max):
                index = torch.where(candidate_instance_labels == i)[0]
                if index.shape[0] == 0:
                    continue
                index_h = index.view(-1, 1).repeat(1, index.shape[0])
                index_v = index.view(1, -1).repeat(index.shape[0], 1)
                merge_gt[index_h, index_v] = 1
            merge_gt[torch.eye(merge_gt.shape[0]).bool()] = 1

            candidate_each_batch_size = torch.bincount(candidate_batch_idxs)
            candidate_merge_loss = F.binary_cross_entropy(merge_pred, merge_gt, reduction='sum') / (candidate_each_batch_size**2).sum()
            candidate_merge_loss += dice_loss(merge_pred.squeeze(), merge_gt.squeeze(), is_sigmoid=False)

        losses['candidate_merge_loss'] = candidate_merge_loss

        return losses

    @force_fp32(apply_to=('pred_label', 'pred_score', 'pred_mask'))
    def query_instance_layer_loss(self, pred_label, pred_score, pred_mask, instances3D_label):
        '''
            pred_label: (layer, num_query, label_num+1)
            pred_score: (layer, num_query, 1)
            pred_mask: (layer, num_query, num_point)
        '''
        loss_out = {}

        layer_num = len(pred_label)
        self.layer_num = layer_num

        pred_label = torch.cat(pred_label, dim=0)
        pred_score = torch.cat(pred_score, dim=0)
        pred_mask = torch.cat(pred_mask, dim=0)

        tgt_class, tgt_mask, weights = instances3D_label
        tgt_class = tgt_class.repeat(layer_num)
        tgt_mask = tgt_mask.repeat(layer_num, 1)
        weights = weights.repeat(layer_num, 1)

        vaild_inds = (tgt_class != tgt_class.new_full((1,), self.ignore_label))

        pred_label = pred_label[vaild_inds]
        tgt_class = tgt_class[vaild_inds]
        pred_score = pred_score[vaild_inds]
        pred_mask = pred_mask[vaild_inds]
        tgt_mask = tgt_mask[vaild_inds]
        weights = weights[vaild_inds]

        # class loss
        # class_loss = F.cross_entropy(pred_label, tgt_class, ignore_index=self.ignore_label)
        class_loss = F.cross_entropy(pred_label, tgt_class.long(), weight=torch.tensor([*self.query_cfg.inst_weight, 1.0], device=pred_label.device), reduction='sum') / (pred_label.shape[0] / layer_num)

        loss_out['cls_loss'] = class_loss * self.query_loss_weight[0]

        # score loss
        with torch.no_grad():
            tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)   # (num_inst, 1)

        filter_id, _ = torch.where(tgt_score > 0.5)
        if filter_id.numel():
            tgt_score = tgt_score[filter_id]
            pred_score = pred_score[filter_id]
        # score_loss = F.mse_loss(pred_score, tgt_score)          # iou的score预测loss
        score_loss = F.mse_loss(pred_score, tgt_score, reduction='sum') / (pred_score.shape[0] / layer_num)

        # mask loss
        # mask_bce_loss = F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
        # mask_dice_loss = dice_loss(pred_mask, tgt_mask.float())    # 类似于计算iou，但要使得loss最小
        mask_bce_loss = F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float(), weight=weights, reduction='sum') / (pred_mask.shape[0] * pred_mask.shape[1] / layer_num)
        mask_dice_loss = dice_loss(pred_mask, tgt_mask.float(), weights=weights, is_mean=False) / (pred_mask.shape[0] / layer_num)

        loss_out['mask_bce_loss'] = mask_bce_loss * self.query_loss_weight[1]
        loss_out['mask_dice_loss'] = mask_dice_loss * self.query_loss_weight[2]
        loss_out['score_loss'] = score_loss * self.query_loss_weight[3]

        return loss_out

    def parse_losses(self, losses):
        loss = sum(v for v in losses.values())
        # losses['loss'] = loss
        losses_out = {}
        for loss_name, loss_value in losses.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))

            if loss_name in 'cls_loss,mask_bce_loss,mask_dice_loss,score_loss':
                losses_out['mean_' + loss_name] = loss_value.item() / self.layer_num
            else:
                losses_out[loss_name] = loss_value.item()
        losses_out['loss'] = loss

        ''' '''
        losses_out['vaild_fore'] = self.invaild_fore
        ''' '''

        return loss, losses_out

    @cuda_cast
    def forward_test(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                     semantic_labels, instance_labels, pt_offset_labels, spatial_shape, batch_size,
                     scan_ids, **kwargs):

        output_feats, sparse_batch_idxs, sparse_id, sparse_coords, _ = self.forward_backbone(
            torch.cat((feats, coords_float), 1), p2v_map, voxel_coords, spatial_shape, batch_size
        )

        candidate_scores, semantic_scores, pt_offsets, _, (candidate_beforeMerge_id, candidate_afterMerge_id), output_feats, query_feats = self.forward_point_wise(
            output_feats, v2p_map, sparse_batch_idxs, sparse_coords, sparse_id
        )


        # import open3d as o3d
        # import numpy as np
        # import matplotlib.pyplot as plt
        # np_coords_float = coords_float.cpu().numpy()
        # # np_feats = output_feats.features[v2p_map.long()].cpu().numpy().max(-1)
        # # np_feats = np.linalg.norm(pt_offsets.cpu().numpy(), axis=-1)
        # np_feats = candidate_scores.cpu().numpy()
        # normalized_np_feats = (np_feats - np.min(np_feats)) / (np.max(np_feats) - np.min(np_feats))
        # cmap = plt.get_cmap('jet')
        # rgb_values = cmap(normalized_np_feats)[:, :-1]
        # pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np_coords_float))
        # pcd.colors = o3d.utility.Vector3dVector(rgb_values)
        # o3d.visualization.draw_geometries([pcd])


        inst_label = instance_labels[instance_labels != self.ignore_label]
        inst_num = torch.unique(inst_label).shape[0]

        semantic_preds = semantic_scores.max(1)[1]
        ret = dict(
            scan_id=scan_ids[0],
            coords_float=coords_float.cpu().numpy(),
            semantic_preds=semantic_preds.cpu().numpy(),
            semantic_labels=semantic_labels.cpu().numpy(),
            offset_preds=pt_offsets.cpu().numpy(),
            offset_labels=pt_offset_labels[:, 0:3].cpu().numpy(),
            instance_labels=instance_labels.cpu().numpy(),
            instance_nums=inst_num,
            candidate_preds=candidate_scores.cpu().numpy(),
            candidate_labels=pt_offset_labels[:, 3].cpu().numpy(),
            candidate_beforeMerge_id=candidate_beforeMerge_id.cpu().numpy(),
            candidate_afterMerge_id=candidate_afterMerge_id.cpu().numpy()
        )
        if not self.semantic_only:
            # if not self.query_cfg.use_feat_query:
            #     query_feats = self.query.weight.unsqueeze(0)
            prediction = self.query_instance_layer(
                output_feats[0], query_feats[0],
                v2p_map.long(), sparse_batch_idxs, 0)

            pred_instances = self.get_instances(scan_ids[0], prediction['labels'][-1], prediction['masks'][-1], prediction['scores'][-1])
            gt_instances = self.get_gt_instances(semantic_labels, instance_labels)
            ret.update(dict(pred_instances=pred_instances, gt_instances=gt_instances))
        return ret

    def forward_backbone(self, feats, p2v_map, voxel_coords, spatial_shape, batch_size):
        voxel_feats = voxelization(feats, p2v_map)

        sparse_id = p2v_map[:, 1].squeeze().long()

        feats = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        output = self.input_conv(feats)
        output, aux_out = self.unet(output)
        output = self.output_layer(output)

        return output, output.indices[:, 0].int(), sparse_id, voxel_feats[:, 3:].contiguous(), aux_out

    def forward_point_wise(self, input_features, v2p_map, sparse_batch_idxs, sparse_coords, sparse_id):
        input = input_features.features

        # semantic
        semantic_scores = self.semantic_linear(input)

        # offset
        pt_offsets = self.offset_linear(input)   # (Nv, 3)

        # candidate
        candidate_scores = self.candidate_linear(self.catFeat_linear(torch.cat([input, pt_offsets], dim=-1))).squeeze()   # (Nv)

        # centroid select
        sparse_batch_size = sparse_batch_idxs.max() + 1
        batch_offsets = self.get_batch_offsets(sparse_batch_idxs, sparse_batch_size).cpu()

        semantic_preds = semantic_scores.max(1)[1].int()
        heatmap = torch.clamp(candidate_scores.detach(), 1e-6, None)
        topk_idxs, k_foreground_idxs = topk_nms(
            heatmap, batch_offsets, sparse_coords, semantic_preds,
            self.candidate_nms_R, self.candidate_nms_thres, self.candidate_nms_localThres,
            self.candidate_nms_N * sparse_batch_size, self.candidate_nms_maxActive_f
        )
        topk_idxs = topk_idxs.cuda()
        k_foreground_idxs = k_foreground_idxs.cuda().long()

        # make candidate feat
        catfeat = self.query_catFeat_linear(torch.cat([input, pt_offsets + sparse_coords], dim=-1))

        candidate_batch_idxs = sparse_batch_idxs[topk_idxs]
        candidate_feats = catfeat[topk_idxs]

        k_num = topk_idxs.shape[0]
        self.foreground_mask = []
        for i in range(k_num):
            mask = torch.where(k_foreground_idxs[:, 0] == i)[0]
            mask = k_foreground_idxs[mask, 1]
            if mask.shape[0] != 0:
                self.foreground_mask.append(sparse_id[mask])
                foreground_feats = catfeat[mask]
                if mask.shape[0] > 50:
                    foreground_scores = candidate_scores[mask]
                    sort_idx = foreground_scores.argsort(descending=True)
                    foreground_feats = foreground_feats[sort_idx[:50]]
                candidate_feats[i] = torch.cat([candidate_feats[i].unsqueeze(0), foreground_feats], dim=0).mean(0)

        # centroid merge
        merge_pred, merge_index = self.candidate_merge(candidate_feats, candidate_scores[topk_idxs], sparse_coords[topk_idxs] + pt_offsets[topk_idxs], candidate_batch_idxs)

        # tiny
        inst_feat = self.tiny_unet_inputlayer(input_features.replace_feature(catfeat.detach().requires_grad_()))
        inst_feat = self.tiny_unet(inst_feat)
        inst_feat = self.tiny_unet_outputlayer(inst_feat).features



        # import open3d as o3d
        # import numpy as np
        # import matplotlib.pyplot as plt
        # np_coords_float = sparse_coords.cpu().numpy()
        # np_feats = inst_feat.cpu().numpy().mean(-1)
        # # np_feats = np.linalg.norm(pt_offsets.cpu().numpy(), axis=-1)
        # # np_feats = candidate_scores.cpu().numpy()
        # normalized_np_feats = (np_feats - np.min(np_feats)) / (np.max(np_feats) - np.min(np_feats))
        # cmap = plt.get_cmap('jet')
        # rgb_values = cmap(normalized_np_feats)[:, :-1]
        # pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np_coords_float))
        # pcd.colors = o3d.utility.Vector3dVector(rgb_values)
        # o3d.visualization.draw_geometries([pcd])



        candidate_feats = inst_feat[topk_idxs]

        k_num = topk_idxs.shape[0]
        for i in range(k_num):
            mask = torch.where(k_foreground_idxs[:, 0] == i)[0]
            mask = k_foreground_idxs[mask, 1]
            if mask.shape[0] != 0:
                foreground_feats = inst_feat[mask]
                if mask.shape[0] > 50:
                    foreground_scores = candidate_scores[mask]
                    sort_idx = foreground_scores.argsort(descending=True)
                    foreground_feats = foreground_feats[sort_idx[:50]]
                # candidate_feats[i] = torch.cat([candidate_feats[i].unsqueeze(0), foreground_feats], dim=0).mean(0)
                candidate_feats[i] = foreground_feats.mean(0) * 0.7 + candidate_feats[i] * 0.3

        candidate_feats = candidate_feats[merge_index]
        candidate_batch_idxs = candidate_batch_idxs[merge_index]

        query_feats = []
        batch_feats = []
        if not self.semantic_only:
            candidate_batch_size = candidate_batch_idxs.max() + 1
            for i in range(candidate_batch_size):
                query_feats.append(candidate_feats[torch.where(i == candidate_batch_idxs)])

            sparse_batch_size = sparse_batch_idxs.max() + 1
            for i in range(sparse_batch_size):
                batch_feats.append(inst_feat[torch.where(i == sparse_batch_idxs)])

        return candidate_scores[v2p_map.long()], semantic_scores[v2p_map.long()], pt_offsets[v2p_map.long()], merge_pred, (sparse_id[topk_idxs], sparse_id[topk_idxs][merge_index]), batch_feats, query_feats

# ==============================================
# ****************** Function ******************
# ==============================================
    def get_batch_offsets(self, batch_idxs, bs):
        batch_offsets = torch.zeros(bs + 1).int().cuda()
        for i in range(bs):
            batch_offsets[i + 1] = batch_offsets[i] + torch.where(batch_idxs == i)[0].shape[0]
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets

    def get_gt_instances(self, semantic_labels, instance_labels):
        """Get gt instances for evaluation."""
        # convert to evaluation format 0: ignore, 1->N: valid
        label_shift = self.semantic_classes - self.instance_classes
        semantic_labels = semantic_labels - label_shift + 1
        semantic_labels[semantic_labels < 0] = 0
        instance_labels += 1
        ignore_inds = instance_labels < 0
        # scannet encoding rule
        gt_ins = semantic_labels * 1000 + instance_labels
        gt_ins[ignore_inds] = 0
        gt_ins = gt_ins.cpu().numpy()
        return gt_ins

    @force_fp32(apply_to=('pred_labels', 'pred_masks', 'pred_scores'))
    def get_instances(self, scan_id, pred_labels, pred_masks, pred_scores):
        scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        # scores *= pred_scores
        labels = torch.arange(self.instance_classes, device=scores.device).unsqueeze(0).repeat(pred_labels.shape[0], 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(scores.shape[0] * scores.shape[1], sorted=False)
        labels = labels[topk_idx]   # 根据scores排序后取对应的labels
        labels += 1

        topk_idx = torch.div(topk_idx, self.instance_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        # mask_pred before sigmoid()
        mask_pred = (mask_pred > self.test_cfg.mask_score_thr).float()  # [n_p, M], 以0为界限
        mask_scores = (mask_pred_sigmoid * mask_pred).sum(1) / (mask_pred.sum(1) + 1e-6)
        scores = scores * mask_scores

        # score_thr
        score_mask = scores > self.test_cfg.cls_score_thr
        scores = scores[score_mask]  # (n_p,)
        labels = labels[score_mask]  # (n_p,)
        mask_pred = mask_pred[score_mask]  # (n_p, N)

        # npoint thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.min_npoint
        scores = scores[npoint_mask]  # (n_p,)
        labels = labels[npoint_mask]  # (n_p,)
        mask_pred = mask_pred[npoint_mask]  # (n_p, N)

        cls_pred = labels.cpu().numpy()
        score_pred = scores.cpu().numpy()
        mask_pred = mask_pred.cpu().numpy()

        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_id
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            pred_instances.append(pred)

        return pred_instances

    def candidate_merge(self, candidate_feats, candidate_scores, pos, batch_idx, merge_thre=0.5):
        '''
        :param candidate_feats: (N', d'), float, cuda
        :param candidate_scores: (N'), float
        :param pos: (N', 3), float, cuda
        :param batch_idx: (N'), float, int
        :param merge_thre: int, cuda
        :return: merge_pred: (I, I), float, cuda
        :return: ins_map: (I), int, cuda
        '''
        if candidate_feats.shape[0] == 0 or candidate_feats.shape[0] == 1:
            return torch.tensor([], dtype=torch.float32, device=candidate_feats.device), torch.tensor([], dtype=torch.long, device=candidate_feats.device)

        input_pos = torch.cat((candidate_feats, pos), dim=1)  # I * (d'+3)
        batch_mask = batch_idx.unsqueeze(1) - batch_idx.unsqueeze(0)  # batch mask (I * I)

        input_diff = torch.clamp(torch.abs(input_pos.unsqueeze(1) - input_pos.unsqueeze(0)), min=1e-6).contiguous().view(-1, input_pos.shape[-1]).contiguous()  # I * I * (d'+3)
        merge_pred = self.mergeCandidate_linear(input_diff).view(input_pos.shape[0], input_pos.shape[0], 1).contiguous().sigmoid()  # I * I * 1
        merge_pred = torch.where(batch_mask.unsqueeze(-1) != 0, torch.zeros_like(merge_pred, device=merge_pred.device), merge_pred)

        merge_score = merge_pred.clone().detach()

        # (I), instance_id of each candidate
        ins_map = torch.arange(candidate_feats.shape[0])

        ins_num = ins_map.shape[0]  # I
        merge_score[torch.eye(ins_num).bool()] = 0

        # greedy aggregation
        while merge_score.max() > merge_thre:
            index = merge_score.argmax()  # max score
            # i = index // ins_num  # candidate i
            i = torch.div(index, ins_num, rounding_mode='floor')
            j = index % ins_num  # candidate j

            # group i, candidates with the same instance_id as candidate i
            i_group = torch.where(ins_map[:] == ins_map[i])[0]
            # group j, candidates with the same instance_id as candidate j
            j_group = torch.where(ins_map[:] == ins_map[j])[0]
            new_group = torch.cat((i_group, j_group), dim=0)  # merged group

            new_group_h = new_group.view(-1, 1).repeat(1, new_group.shape[0])
            new_group_v = new_group.view(1, -1).repeat(new_group.shape[0], 1)
            # set scores within the new group to 0
            merge_score[new_group_h, new_group_v] = 0

            ins_map_tmp = ins_map.clone()
            ins_map[new_group] = min(ins_map_tmp[i], ins_map_tmp[j])  # update ins_map

        _, ins_map = torch.unique(ins_map, return_inverse=True)
        ins_map_max = ins_map.max() + 1
        select_index = []
        for im in range(ins_map_max):
            mask = torch.where(ins_map == im)[0]
            if mask.shape[0] > 1:
                index = candidate_scores[mask].argmax()
                select_index.append(mask[index])
            else:
                select_index.append(mask[0])

        select_index = torch.tensor(select_index, dtype=torch.long, device=candidate_feats.device)

        return merge_pred, select_index

    def get_instance3D(self, semantic_labels, instance_labels, candidate_afterMerge_id, batch_mask):
        candidate_sem_label = semantic_labels[candidate_afterMerge_id]
        candidate_inst_label = instance_labels[candidate_afterMerge_id]
        batch_instance_labels = instance_labels[batch_mask]

        num_points = batch_instance_labels.shape[0]
        gt_masks = []
        weights = []
        for i, cil in enumerate(candidate_inst_label):
            idx = torch.where(cil == batch_instance_labels)[0]
            gt_mask = torch.zeros(num_points, device=semantic_labels.device)
            gt_mask[idx] = 1
            gt_masks.append(gt_mask)

            weight = torch.ones(num_points, device=semantic_labels.device) * (0 if candidate_sem_label[i] == self.ignore_label else self.query_cfg.inst_weight[candidate_sem_label[i]])
            weights.append(weight)

        if gt_masks:
            gt_masks = torch.stack(gt_masks, dim=0)         # (num_insts, num_points)

        if weights:
            weights = torch.stack(weights, dim=0)

        return (candidate_sem_label, gt_masks, weights)
