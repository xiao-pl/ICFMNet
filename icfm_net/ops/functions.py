import torch
from torch.autograd import Function

from . import ops


class GetMaskIoUOnCluster(Function):

    @staticmethod
    def forward(ctx, proposals_idx, proposals_offset, instance_labels, instance_pointnum):
        '''
        :param ctx:
        :param proposals_idx: (sumNPoint), int
        :param proposals_offset: (nProposal + 1), int
        :param instance_labels: (N), long, 0~total_nInst-1, -100
        :param instance_pointnum: (total_nInst), int
        :param mask_scores_sigmoid: (sumNPoint), float
        :param mode: int, mode = 1 if cal IoU based on mask else mode = 0

        :return: proposals_iou: (nProposal, total_nInst), float
        :return mask_label:
        '''

        nInstance = instance_pointnum.size(0)
        nProposal = proposals_offset.size(0) - 1
        proposals_iou = torch.cuda.FloatTensor(nProposal, nInstance).zero_()

        assert proposals_idx.is_contiguous() and proposals_idx.is_cuda
        assert proposals_offset.is_contiguous() and proposals_offset.is_cuda
        assert instance_labels.is_contiguous() and instance_labels.is_cuda
        assert instance_pointnum.is_contiguous() and instance_pointnum.is_cuda

        ops.get_mask_iou_on_cluster(proposals_idx, proposals_offset, instance_labels,
                                    instance_pointnum, proposals_iou, nInstance, nProposal)

        return proposals_iou

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


get_mask_iou_on_cluster = GetMaskIoUOnCluster.apply


class GetMaskIoUOnPred(Function):

    @staticmethod
    def forward(ctx, proposals_idx, proposals_offset, instance_labels, instance_pointnum,
                mask_scores_sigmoid):
        '''
        :param ctx:
        :param proposals_idx: (sumNPoint), int
        :param proposals_offset: (nProposal + 1), int
        :param instance_labels: (N), long, 0~total_nInst-1, -100
        :param instance_pointnum: (total_nInst), int
        :param mask_scores_sigmoid: (sumNPoint), float
        :param mode: int, mode = 1 if cal IoU based on mask else mode = 0

        :return: proposals_iou: (nProposal, total_nInst), float
        :return mask_label:
        '''

        nInstance = instance_pointnum.size(0)
        nProposal = proposals_offset.size(0) - 1
        proposals_iou = torch.cuda.FloatTensor(nProposal, nInstance).zero_()

        assert proposals_idx.is_contiguous() and proposals_idx.is_cuda
        assert proposals_offset.is_contiguous() and proposals_offset.is_cuda
        assert instance_labels.is_contiguous() and instance_labels.is_cuda
        assert instance_pointnum.is_contiguous() and instance_pointnum.is_cuda
        assert mask_scores_sigmoid.is_contiguous() and mask_scores_sigmoid.is_cuda

        ops.get_mask_iou_on_pred(proposals_idx, proposals_offset, instance_labels,
                                 instance_pointnum, proposals_iou, nInstance, nProposal,
                                 mask_scores_sigmoid)

        return proposals_iou

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


get_mask_iou_on_pred = GetMaskIoUOnPred.apply


class GetMaskLabel(Function):

    @staticmethod
    def forward(ctx, proposals_idx, proposals_offset, instance_labels, instance_cls,
                instance_pointnum, proposals_iou, iou_thr):
        '''
        :param ctx:
        :param proposals_idx: (sumNPoint), int
        :param proposals_offset: (nProposal + 1), int
        :param instance_labels: (N), long, 0~total_nInst-1, -100
        :param mask_scores_sigmoid: (sumNPoint), float
        :param mode: int, mode = 1 if cal IoU based on mask else mode = 0

        :return: proposals_iou: (nProposal, total_nInst), float
        :return mask_label:
        '''

        nInstance = instance_pointnum.size(0)
        nProposal = proposals_offset.size(0) - 1
        mask_label = torch.cuda.FloatTensor(proposals_idx.shape).zero_() - 1.

        assert proposals_iou.is_contiguous() and proposals_iou.is_cuda
        assert proposals_idx.is_contiguous() and proposals_idx.is_cuda
        assert proposals_offset.is_contiguous() and proposals_offset.is_cuda
        assert instance_labels.is_contiguous() and instance_labels.is_cuda
        assert instance_cls.is_contiguous() and instance_cls.is_cuda

        ops.get_mask_label(proposals_idx, proposals_offset, instance_labels, instance_cls,
                           proposals_iou, nInstance, nProposal, iou_thr, mask_label)

        return mask_label

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


get_mask_label = GetMaskLabel.apply


class Voxelization_Idx(Function):

    @staticmethod
    def forward(ctx, coords, batchsize, mode=4):
        '''
        :param ctx:
        :param coords:  long (N, dimension + 1) or (N, dimension) dimension = 3
        :param batchsize
        :param mode: int 4=mean
        :param dimension: int
        :return: output_coords:  long (M, dimension + 1) (M <= N)
        :return: output_map: int M * (maxActive + 1)
        :return: input_map: int N
        '''
        assert coords.is_contiguous()
        N = coords.size(0)
        output_coords = coords.new()

        input_map = torch.IntTensor(N).zero_()
        output_map = input_map.new()

        ops.voxelize_idx(coords, output_coords, input_map, output_map, batchsize, mode)
        return output_coords, input_map, output_map

    @staticmethod
    def backward(ctx, a=None, b=None, c=None):
        return None


voxelization_idx = Voxelization_Idx.apply


class Voxelization(Function):

    @staticmethod
    def forward(ctx, feats, map_rule, mode=4):
        '''
        :param ctx:
        :param map_rule: cuda int M * (maxActive + 1)
        :param feats: cuda float N * C
        :return: output_feats: cuda float M * C
        '''
        assert map_rule.is_contiguous()
        assert feats.is_contiguous()
        N, C = feats.size()
        M = map_rule.size(0)
        maxActive = map_rule.size(1) - 1

        output_feats = torch.cuda.FloatTensor(M, C).zero_()

        ctx.for_backwards = (map_rule, mode, maxActive, N)

        ops.voxelize_fp(feats, output_feats, map_rule, mode, M, maxActive, C)
        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        map_rule, mode, maxActive, N = ctx.for_backwards
        M, C = d_output_feats.size()

        d_feats = torch.cuda.FloatTensor(N, C).zero_()

        ops.voxelize_bp(d_output_feats.contiguous(), d_feats, map_rule, mode, M, maxActive, C)
        return d_feats, None, None


voxelization = Voxelization.apply


class BallQueryBatchP(Function):

    @staticmethod
    def forward(ctx, coords, batch_idxs, batch_offsets, radius, meanActive):
        '''
        :param ctx:
        :param coords: (n, 3) float
        :param batch_idxs: (n) int
        :param batch_offsets: (B+1) int
        :param radius: float
        :param meanActive: int
        :return: idx (nActive), int
        :return: start_len (n, 2), int
        '''

        n = coords.size(0)

        assert coords.is_contiguous() and coords.is_cuda
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda
        assert batch_offsets.is_contiguous() and batch_offsets.is_cuda

        while True:
            idx = torch.cuda.IntTensor(n * meanActive).zero_()
            start_len = torch.cuda.IntTensor(n, 2).zero_()
            nActive = ops.ballquery_batch_p(coords, batch_idxs, batch_offsets, idx, start_len, n,
                                            meanActive, radius)
            if nActive <= n * meanActive:
                break
            meanActive = int(nActive // n + 1)
        idx = idx[:nActive]

        return idx, start_len

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None


ballquery_batch_p = BallQueryBatchP.apply


class BFSCluster(Function):

    @staticmethod
    def forward(ctx, cluster_numpoint_mean, ball_query_idxs, start_len, threshold, class_id):
        '''
        :param ctx:
        :param ball_query_idxs: (nActive), int
        :param start_len: (N, 2), int
        :return: cluster_idxs:  int (sumNPoint, 2), dim 0 for cluster_id, dim 1 for point idxs in N
        :return: cluster_offsets: int (nCluster + 1)
        '''

        N = start_len.size(0)
        assert cluster_numpoint_mean.is_contiguous()
        assert ball_query_idxs.is_contiguous()
        assert start_len.is_contiguous()

        cluster_idxs = ball_query_idxs.new()
        cluster_offsets = ball_query_idxs.new()

        ops.bfs_cluster(cluster_numpoint_mean, ball_query_idxs, start_len, cluster_idxs,
                        cluster_offsets, N, threshold, class_id)

        return cluster_idxs, cluster_offsets

    @staticmethod
    def backward(ctx, a=None):
        return None


bfs_cluster = BFSCluster.apply


class GlobalAvgPool(Function):

    @staticmethod
    def forward(ctx, feats, proposals_offset):
        '''
        :param ctx:
        :param feats: (sumNPoint, C) float
        :param proposals_offset: (nProposal + 1) int
        :return: output_feats (nProposal, C) float
        '''
        nProposal = proposals_offset.size(0) - 1
        sumNPoint, C = feats.size()

        assert feats.is_contiguous()
        assert proposals_offset.is_contiguous()

        output_feats = torch.cuda.FloatTensor(nProposal, C).zero_()

        ops.global_avg_pool_fp(feats, proposals_offset, output_feats, nProposal, C)

        ctx.for_backwards = (proposals_offset, sumNPoint)

        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        nProposal, C = d_output_feats.size()

        proposals_offset, sumNPoint = ctx.for_backwards

        d_feats = torch.cuda.FloatTensor(sumNPoint, C).zero_()

        ops.global_avg_pool_bp(d_feats, proposals_offset, d_output_feats.contiguous(), nProposal, C)

        return d_feats, None


global_avg_pool = GlobalAvgPool.apply


class SecMean(Function):

    @staticmethod
    def forward(ctx, inp, offsets):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1
        C = inp.size(1)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        out = torch.cuda.FloatTensor(nProposal, C).zero_()

        ops.sec_mean(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_mean = SecMean.apply


class SecMin(Function):

    @staticmethod
    def forward(ctx, inp, offsets):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1
        C = inp.size(1)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        out = torch.cuda.FloatTensor(nProposal, C).zero_()

        ops.sec_min(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_min = SecMin.apply


class SecMax(Function):

    @staticmethod
    def forward(ctx, inp, offsets):
        '''
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        '''
        nProposal = offsets.size(0) - 1
        C = inp.size(1)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        out = torch.cuda.FloatTensor(nProposal, C).zero_()

        ops.sec_max(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_max = SecMax.apply


class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset, new_offset):
        """
        input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist2: (m, nsample)
        """
        if new_xyz is None:
            new_xyz = xyz
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        ops.knnquery_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2)
        return idx, None    # torch.sqrt(dist2)

    @staticmethod
    def backward(ctx, a=None):
        return None, None


knnquery = KNNQuery.apply


def knn(ref, query, k):
    d, i = ops.knn(ref, query, k)
    i -= 1
    return d, i


class KNN(Function):
    """
    'https://github.com/unlimblue/KNN_CUDA'

    if transpose_mode is True,
        ref   is Tensor [bs x nr x dim]
        query is Tensor [bs x nq x dim]

        return
            dist is Tensor [bs x nq x k]
            indx is Tensor [bs x nq x k]
    else
        ref   is Tensor [bs x dim x nr]
        query is Tensor [bs x dim x nq]

        return
            dist is Tensor [bs x k x nq]
            indx is Tensor [bs x k x nq]
    """
    # @staticmethod
    # def forward(ctx, ref, query, k, transpose_mode=False):
    #     _t = transpose_mode
    #     assert ref.size(0) == query.size(0), "ref.shape={} != query.shape={}".format(ref.shape, query.shape)

    #     batch_size = ref.size(0)
    #     D, I = [], []
    #     for bi in range(batch_size):
    #         r, q = _T(ref[bi], _t), _T(query[bi], _t)
    #         d, i = knn(r.float(), q.float(), k)
    #         d, i = _T(d, _t), _T(i, _t)
    #         D.append(d)
    #         I.append(i)
    #     D = torch.stack(D, dim=0)
    #     I = torch.stack(I, dim=0)
    #     return D, I
    @staticmethod
    def forward(ctx, xyz, offset, nsample):
        assert xyz.is_contiguous()

        idx = []

        for iii in range(offset.shape[0]):
            idx_offset = 0 if iii == 0 else offset[iii - 1]

            knn_data = xyz[idx_offset:offset[iii], :]

            knn_data = knn_data.transpose(0, 1).contiguous()
            _, i = knn(knn_data.float(), knn_data.float(), nsample + 1)
            idx.append(i.transpose(0, 1).contiguous().int()[:, 1:] + idx_offset)
        idx = torch.cat(idx)

        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None


knn_cuda = KNN.apply


class KNN_V2(Function):
    @staticmethod
    def forward(ctx, xyz, offset, nsample):
        assert xyz.is_contiguous()

        m = xyz.shape[0]

        nsample += 1
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()

        ops.knn_v2(nsample, xyz, offset, idx, dist2)

        return idx[:, 1:].contiguous()

    @staticmethod
    def backward(ctx, a=None):
        return None, None


knn_cuda_v2 = KNN_V2.apply


import open3d as o3d


def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), feat: (n, c), idx: (m, nsample), offset: (b), new_offset: (b)
    output: new_feat: (m, c+3, nsample), grouped_idx: (m, nsample)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        # idx = knnquery(nsample, xyz, new_xyz, offset, new_offset)[0]    # (m, nsample)

        try:
            # idx = knn_cuda(xyz, offset, nsample)    # (m, nsample)
            idx = knn_cuda_v2(xyz, offset, nsample)    # (m, nsample)
        except Exception as e:
            print("[queryandgroup]: " + str(e))
            idx = []
            for iii in range(offset.shape[0]):
                idx_offset = 0 if iii == 0 else offset[iii - 1]

                knn_data = xyz[idx_offset:offset[iii], :].cpu().numpy()

                pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(knn_data))
                pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                pcd_idx = []
                for kd in knn_data:
                    [_, knn_idx, _] = pcd_tree.search_knn_vector_3d(kd, nsample + 1)
                    pcd_idx.append(torch.cuda.IntTensor(knn_idx[1:]) + idx_offset)
                pcd_idx = torch.cat(pcd_idx)
                idx.append(pcd_idx)
            idx = torch.cat(idx)

    _, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3)       # (m, nsample, 3)
    # grouped_xyz = grouping(xyz, idx)      # (m, nsample, 3)
    grouped_xyz -= new_xyz.unsqueeze(1)     # (m, nsample, 3)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)     # (m, nsample, c)
    # grouped_feat = grouping(feat, idx) # (m, nsample, c)

    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1)               # (m, nsample, 3+c)
    else:
        return grouped_feat


class TopKNMS(Function):
    @staticmethod
    def forward(ctx, heatmap, batch_offset, coord, label,
                R=0.3, thres=0.3, local_thres=0.5,
                K=100, maxActive_f=1000):
        '''
        :param ctx:
        :param input: (n) int
        :param batch_offset: (B+1) int
        :param coord: (n, 3) float
        :param R: float
        :param meanActive: int
        :return: idx (nActive), int
        :return: start_len (n, 2), int
        '''

        n = coord.size(0)

        assert heatmap.is_contiguous() and heatmap.is_cuda
        assert coord.is_contiguous() and coord.is_cuda
        assert label.is_contiguous() and label.is_cuda
        assert batch_offset.is_contiguous() and not batch_offset.is_cuda

        topk_idxs = torch.zeros(K).int()
        # sizes = torch.zeros(K).int()        # 每个topk的foreground数量
        k_foreground = torch.zeros((K*maxActive_f, 2)).int()    # (索引, idx), 对应topk_idxs
        # k_background = torch.zeros((K*maxActive_b, 2)).int()

        # if input.min() < 1e-6 and input.min() != 0:
        #     scale = 1e-6 / input.min()
        #     thres = thres * scale
        #     input = input * scale

        # thres = max(1e-6, thres)

        nActive = ops.topk_nms(coord, heatmap, batch_offset, label,
                               R, thres, local_thres, K,
                               topk_idxs, maxActive_f, k_foreground)

        topk_idxs = topk_idxs[:nActive].long()
        # sizes = sizes[:nActive]

        return topk_idxs, k_foreground

    @staticmethod
    def backward(ctx, a=None, b=None, c=None, d=None):
        return None, None, None, None, None


topk_nms = TopKNMS.apply
