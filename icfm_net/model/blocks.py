from collections import OrderedDict

import spconv.pytorch as spconv
import numpy as np
import torch
from spconv.pytorch.modules import SparseModule
from torch import nn, einsum
from einops import repeat
from ..util import exists, max_value, batched_index_select
from ..ops import queryandgroup


class MLP(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels, bias=False))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)


# current 1x1 conv in spconv2x has a bug. It will be removed after the bug is fixed
class Custom1x1Subm3d(spconv.SparseConv3d):

    def forward(self, input):
        features = torch.mm(input.features, self.weight.view(self.out_channels, self.in_channels).T)
        if self.bias is not None:
            features += self.bias
        out_tensor = spconv.SparseConvTensor(features, input.indices, input.spatial_shape, input.batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class ResidualBlock(SparseModule):

    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                Custom1x1Subm3d(in_channels, out_channels, kernel_size=1, bias=False))

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels), nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels), nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key))

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape,
                                           input.batch_size)
        output = self.conv_branch(input)
        out_feats = output.features + self.i_branch(identity).features
        output = output.replace_feature(out_feats)

        return output


class UBlock(nn.Module):

    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1, use_transformer=False, transformer_before=False, nsample: int = 16, stop_trans=2):

        super().__init__()

        self.nPlanes = nPlanes

        self.use_transformer = use_transformer
        self.transformer_before = transformer_before
        self.stop_trans = stop_trans

        blocks = {
            'block{}'.format(i):
            block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1, use_transformer=use_transformer, transformer_before=transformer_before, nsample=nsample - 8, stop_trans=stop_trans)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

        if use_transformer and len(nPlanes) > self.stop_trans:
            self.transformerLayer = PointAttentionLayer(in_planes=nPlanes[0], nsample=np.clip(nsample, 16, None))

    def forward(self, input):

        output = self.blocks(input)

        # if self.transformer_before and self.use_transformer and len(self.nPlanes) > self.stop_trans:
        #     out_feats = self.transformerLayer(output)
        #     output = output.replace_feature(out_feats)

        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)
        aux_out = []
        if len(self.nPlanes) > 1:
            # 下采样
            output_decoder = self.conv(output)
            output_decoder, aux_out = self.u(output_decoder)
            # 上采样
            output_decoder = self.deconv(output_decoder)
            out_feats = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(out_feats)
            output = self.blocks_tail(output)

        if not self.transformer_before and self.use_transformer and len(self.nPlanes) > self.stop_trans:
            out_feats = self.transformerLayer(output)
            output = output.replace_feature(out_feats)
        aux_out.append(output)
        return output, aux_out


class UBlock_multipleDecode(nn.Module):

    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1, use_transformer=False, nsample: int = 16, stop_trans=2):

        super().__init__()

        self.nPlanes = nPlanes

        self.use_transformer = use_transformer
        self.stop_trans = stop_trans

        blocks = {
            'block{}'.format(i):
            block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock_multipleDecode(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1, use_transformer=use_transformer, nsample=nsample - 8, stop_trans=stop_trans)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

        if use_transformer and len(nPlanes) > self.stop_trans:
            self.transformerLayer = PointAttentionLayer(in_planes=nPlanes[0], nsample=np.clip(nsample, 16, None))

    def forward(self, input):

        output = self.blocks(input)

        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)
        aux_out = []
        if len(self.nPlanes) > 1:
            # 下采样
            output_decoder = self.conv(output)
            output_decoder, aux_out = self.u(output_decoder)
            # 上采样
            output_decoder = self.deconv(output_decoder)
            out_feats = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(out_feats)
            output = self.blocks_tail(output)

        if self.use_transformer and len(self.nPlanes) > self.stop_trans:
            out_feats = self.transformerLayer(output)
            output = output.replace_feature(out_feats)

        identity.indice_dict = output.indice_dict
        aux_out.append(identity)
        return output, aux_out


class UBlock_onlyDecode(nn.Module):

    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1, use_transformer=False, nsample: int = 16, stop_trans=2, expend_multiple=None):

        super().__init__()

        self.nPlanes = nPlanes

        self.use_transformer = use_transformer
        self.stop_trans = stop_trans

        self.expend_multiple = expend_multiple
        if expend_multiple is not None:
            self.expend_block = spconv.SparseSequential(
                block(nPlanes[0] // expend_multiple, nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            )

        if len(nPlanes) > 1:

            self.u = UBlock_onlyDecode(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1, use_transformer=use_transformer, nsample=nsample - 8, stop_trans=stop_trans, expend_multiple=expend_multiple)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

        if use_transformer and len(nPlanes) > self.stop_trans:
            self.transformerLayer = PointAttentionLayer(in_planes=nPlanes[0], nsample=np.clip(nsample, 16, None))

    def forward(self, input):

        output = input[-1]
        if self.expend_multiple is not None:
            output = self.expend_block(output)

        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.u(input[:-1])
            # 上采样
            output_decoder = self.deconv(output_decoder)
            out_feats = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(out_feats)
            output = self.blocks_tail(output)

        if self.use_transformer and len(self.nPlanes) > self.stop_trans:
            out_feats = self.transformerLayer(output)
            output = output.replace_feature(out_feats)

        return output


class UBlock_tiney(nn.Module):

    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {
            'block{}'.format(i):
            block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]), nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            self.u = UBlock_tiney(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):

        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)
        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)      # 尺寸缩小
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)    # 反卷积
            out_feats = torch.cat((identity.features, output_decoder.features), dim=1)  # 拼接特征
            output = output.replace_feature(out_feats)
            output = self.blocks_tail(output)
        return output


class Norm(nn.Module):
    '''
    LayerNormalization
    '''

    def __init__(self, feats_nums, eps=1e-6):
        super().__init__()

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(feats_nums))
        self.bias = nn.Parameter(torch.zeros(feats_nums))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class PointTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        pos_mlp_hidden_dim=64,
        attn_mlp_hidden_mult=4,
        num_neighbors=None
    ):
        super().__init__()
        self.num_neighbors = num_neighbors

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim, bias=False),
            nn.BatchNorm1d(pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.BatchNorm1d(dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )

    def forward(self, x, pos, mask=None):
        '''
        x: 特征
        pos: 坐标
        mask: 可用于筛选
        '''
        n, num_neighbors = x.shape[1], self.num_neighbors

        # get queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)   # (b, n, dim)

        # calculate relative positional embeddings
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]   # 计算出每个点与其他点的相对位置差 batch * N * N * (dx, dy, dz)
        rel_pos_emb = self.pos_mlp(rel_pos)         # (b, n, n, dim)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        qk_rel = q[:, :, None, :] - k[:, None, :, :]    # (b, n, n, dim)

        # prepare mask
        if exists(mask):
            mask = mask[:, :, None] * mask[:, None, :]  # (b, n, n)

        # expand values
        v = repeat(v, 'b j d -> b i j d', i=n)  # (b, n, n, dim)

        # determine k nearest neighbors for each point, if specified
        if exists(num_neighbors) and num_neighbors < n:
            rel_dist = rel_pos.norm(dim=-1)     # (b, n, n)

            if exists(mask):
                mask_value = max_value(rel_dist)
                rel_dist.masked_fill_(~mask, mask_value)  # 不要的都填充为最大值

            dist, indices = rel_dist.topk(num_neighbors, largest=False)     # topk 用于返回n个最大/小值和索引   (b, n, num_neighbors)

            v = batched_index_select(v, indices, dim=2)
            qk_rel = batched_index_select(qk_rel, indices, dim=2)
            rel_pos_emb = batched_index_select(rel_pos_emb, indices, dim=2)
            mask = batched_index_select(mask, indices, dim=2) if exists(mask) else None

        # add relative positional embeddings to value
        v = v + rel_pos_emb     # (b, n, n, dim)

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)   # (b, n, n, dim)

        # masking
        if exists(mask):
            mask_value = -max_value(sim)
            sim.masked_fill_(~mask[..., None], mask_value)

        # attention
        attn = sim.softmax(dim=-2)      # (b, n, n, dim)

        # aggregate
        agg = einsum('b i j d, b i j d -> b i d', attn, v)  # (b, n, dim)
        return agg


class PointTransformerLayer_1(nn.Module):
    def __init__(
        self,
        in_planes,
        pos_mlp_hidden_dim=48,
        # share_planes=8,
        attn_mlp_hidden_mult=2,
        nsample=16
    ):
        super().__init__()

        self.in_planes = in_planes
        # self.share_planes = share_planes
        self.nsample = nsample
        # self.linear_q = nn.Linear(in_planes, in_planes)
        # self.linear_k = nn.Linear(in_planes, in_planes)
        # self.linear_v = nn.Linear(in_planes, in_planes)
        self.to_qkv = nn.Linear(in_planes, in_planes * 3, bias=False)
        self.linear_p = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim, bias=False),
            nn.BatchNorm1d(pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, in_planes)
        )
        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(in_planes),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes, in_planes * attn_mlp_hidden_mult, bias=False),
            nn.BatchNorm1d(in_planes * attn_mlp_hidden_mult),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes * attn_mlp_hidden_mult, in_planes)
        )
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, pxo):
        '''
        sum(
            (x_k - x_q + p_r) * (x_v + p_r)
        )
        '''
        p, x, o = pxo  # (n, 3), (n, c), (b)

        # x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)

        # get queries, keys, values
        x_q, x_k, x_v = self.to_qkv(x).chunk(3, dim=-1)   # (n, c)

        # x_k = queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        # x_v = queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        # p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]

        x_k = queryandgroup(self.nsample, p, p, torch.cat((x_k, x_v), -1), None, o, o, use_xyz=True)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        x_k, x_v = x_k.chunk(2, dim=-1)         # (n, nsample, c)

        for i, layer in enumerate(self.linear_p):
            p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)

        w = x_k - x_q.unsqueeze(1) + p_r  # (n, nsample, c)

        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)

        # p_r = self.linear_p(p_r)
        # w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        # w = self.linear_w(w)

        w = self.softmax(w)  # (n, nsample, c // s)

        # n, nsample, c = x_v.shape
        # s = self.share_planes
        # x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)

        x = einsum('n s c, n s c -> n c', (x_v + p_r), w)

        return x


class PointAttentionLayer(nn.Module):

    def __init__(self, in_planes, nsample):
        super().__init__()
        self.self_attn = PointTransformerLayer_1(in_planes=in_planes, nsample=nsample)

        self.norm = nn.LayerNorm(in_planes)

        cat_planes = in_planes * 2

        self.catLayer = nn.Sequential(
            nn.Linear(cat_planes, cat_planes, bias=False),
            nn.BatchNorm1d(cat_planes),
            nn.ReLU(),
            nn.Linear(cat_planes, cat_planes, bias=False),
            nn.BatchNorm1d(cat_planes),
            nn.ReLU(),
            nn.Linear(cat_planes, in_planes)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):
        batch_ids = input.indices[:, 0]
        xyz = input.indices[:, 1:].float()
        feats = input.features

        batch_size = batch_ids.max() + 1
        offset, count = [], 0
        for i in range(batch_size):
            # num = list(batch_ids).count(i)
            num = torch.where(batch_ids == i)[0].shape[0]
            if num != 0:
                count += num
                offset.append(count)
        # offset = torch.cuda.IntTensor(offset)
        offset = torch.tensor(offset, dtype=torch.int32, device=feats.device)

        output = self.self_attn((xyz, feats, offset))
        output = self.norm(self.catLayer(torch.cat((feats, output), dim=-1)))
        return output


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, attn_masks=None, pe=None):
        """
        source (B*N, d_model)
        batch_offsets List[int] (b+1)
        query Tensor (b, n_q, d_model)
        """
        query = self.with_pos_embed(query, pe)

        k = v = source
        if attn_masks is not None:
            output, _ = self.attn(query, k, v, attn_mask=attn_masks)  # (1, 100, d_model)
        else:
            output, _ = self.attn(query, k, v)
        output = self.dropout(output) + query
        output = self.norm(output)
        return output


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, pe=None):
        """
        x Tensor (b, 100, c)
        """
        q = k = self.with_pos_embed(x, pe)
        output, _ = self.attn(q, k, x)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output


class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        output = self.net(x) + x
        output = self.norm(output)
        return output


class query_instance_eachLayer_repeat(nn.Module):

    def __init__(
        self,
        block_channels,
        num_decode,
        num_layer,
        num_class,
        in_channel,
        d_model,
        nhead,
        hidden_dim,
        dropout=0.0,
        activation_fn='relu',
        attn_mask=False,
        use_feat_query=True,
        only_last_loss=False,
        **kwargs
    ):
        super().__init__()

        self.num_decode = num_decode
        self.num_layer = num_layer
        self.attn_mask = attn_mask

        self.use_feat_query = use_feat_query
        self.only_last_loss = only_last_loss
        if use_feat_query:
            self.query_layer = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        else:
            self.query_layer = nn.Sequential(nn.Linear(in_channel, d_model, bias=False), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            # self.pos_query_layer = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU())
            self.pos_query_layer = nn.Sequential(nn.Linear(in_channel, d_model, bias=False), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        # self.input_layer = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.mask_layer = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        self.mask_embed_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        # attention layer
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        self.lin_squeeze = nn.ModuleList([])
        for i in range(num_layer):
            self.cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.self_attn_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
            self.lin_squeeze.append(nn.Linear(block_channels[i], d_model))

        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_class + 1))
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

        self.atten_mask_pooling = spconv.SparseAvgPool3d(kernel_size=2, stride=2)

        for m in [self.mask_embed_head, self.mask_layer, self.out_cls, self.out_score]:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.lin_squeeze:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        for p in self.query_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if not use_feat_query:
            for p in self.pos_query_layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def get_mask(self, query, mask_feats, voxel_coords, spatial_shape, pooling_step):
        pred_mask = torch.einsum('nd,md->nm', query, mask_feats)
        attn_mask = None
        if self.attn_mask:
            attn_masks = spconv.SparseConvTensor(pred_mask.T, voxel_coords.int(), spatial_shape, 1)
            for _ in range(pooling_step):
                attn_masks = self.atten_mask_pooling(attn_masks)

            attn_mask = (attn_masks.features.T.sigmoid() < 0.5).bool()  # (num_querys, num_superpoints)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False    # True的位置不进行attend，weight会置为-inf，False的位置保持不变
            attn_mask = attn_mask.detach()

        return pred_mask, attn_mask

    def prediction(self, query, mask_feats,
                   v2p_map, sparse_batch_idxs, batch_idx, voxel_coords, spatial_shape, pooling_step):
        query = self.out_norm(query)
        mask_embed = self.mask_embed_head(query)
        pred_labels = self.out_cls(query)                                   # (num_querys, num_class+1)
        pred_scores = self.out_score(query)                                 # (num_querys, 1)
        pred_masks, attn_masks = self.get_mask(mask_embed, mask_feats,      # (num_querys, num_superpoints), (num_querys, num_superpoints)
                                               voxel_coords, spatial_shape, pooling_step)

        pred_masks_temp = torch.zeros((pred_masks.shape[0], sparse_batch_idxs.shape[0]), dtype=pred_masks.dtype, device=pred_masks.device)
        pred_masks_temp[:, torch.where(sparse_batch_idxs == batch_idx)] = pred_masks
        pred_masks = pred_masks_temp[:, v2p_map]
        batch_idxs = sparse_batch_idxs[v2p_map]
        pred_masks = pred_masks[:, torch.where(batch_idxs == batch_idx)]

        return pred_labels, pred_scores, pred_masks, attn_masks

    def forward(self, input, query,
                v2p_map, sparse_batch_idxs, batch_idx, voxel_coords, spatial_shape,
                aux_out):
        """
        input [B*M, inchannel]
        """
        prediction_labels = []
        prediction_masks = []
        prediction_scores = []
        pos_query = None
        if self.use_feat_query:
            query = self.query_layer(query)
        else:
            pos_query = query.unsqueeze(0)
            for i, layer in enumerate(self.pos_query_layer):
                pos_query = layer(pos_query.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(pos_query)    # (B, query, c)
            pos_query = pos_query[0]

            # query = torch.zeros_like(pos_query)
            query = query.unsqueeze(0)
            for i, layer in enumerate(self.query_layer):
                query = layer(query.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(query)    # (B, query, c)
            query = query[0]
        # inst_feats = self.input_layer(input)
        mask_feats = self.mask_layer(input)

        select_idx = torch.where(sparse_batch_idxs == batch_idx)[0]
        coord = voxel_coords[select_idx]
        coord[:, 0] = 0

        for _ in range(self.num_decode):
            for i in range(self.num_layer):
                pred_labels, pred_scores, pred_masks, attn_masks = self.prediction(
                    query, mask_feats,
                    v2p_map, sparse_batch_idxs, batch_idx, coord, spatial_shape, self.num_layer - i - 1)
                if not self.only_last_loss:
                    prediction_labels.append(pred_labels)
                    prediction_scores.append(pred_scores)
                    prediction_masks.append(pred_masks)

                inst_feats = aux_out[-(self.num_layer - i)]
                indices = inst_feats.indices[:, 0].int()
                inst_feats = inst_feats.features[torch.where(indices == batch_idx)]
                inst_feats = self.lin_squeeze[self.num_layer - i - 1](inst_feats)

                query = self.cross_attn_layers[i](inst_feats, query, attn_masks, pos_query)
                query = self.self_attn_layers[i](query, pos_query)
                query = self.ffn_layers[i](query)

            if self.only_last_loss:
                pred_labels, pred_scores, pred_masks, _ = self.prediction(
                    query, mask_feats,
                    v2p_map, sparse_batch_idxs, batch_idx, coord, spatial_shape, 0)
                prediction_labels.append(pred_labels)
                prediction_scores.append(pred_scores)
                prediction_masks.append(pred_masks)

        if not self.only_last_loss:
            pred_labels, pred_scores, pred_masks, _ = self.prediction(
                query, mask_feats,
                v2p_map, sparse_batch_idxs, batch_idx, coord, spatial_shape, 0)
            prediction_labels.append(pred_labels)
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)

        return dict(
            labels=prediction_labels,
            scores=prediction_scores,
            masks=prediction_masks,
        )


class query_instance_eachLayer_once_finalRepeat(nn.Module):

    def __init__(
        self,
        block_channels,
        num_decode,
        num_layer,
        num_class,
        in_channel,
        d_model,
        nhead,
        hidden_dim,
        dropout=0.0,
        activation_fn='relu',
        attn_mask=False,
        use_feat_query=True,
        only_last_loss=False,
        **kwargs
    ):
        super().__init__()

        self.num_decode = num_decode
        self.num_layer = num_layer
        self.attn_mask = attn_mask

        self.use_feat_query = use_feat_query
        self.only_last_loss = only_last_loss
        if use_feat_query:
            self.query_layer = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        else:
            self.query_layer = nn.Sequential(nn.Linear(in_channel, d_model, bias=False), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            # self.pos_query_layer = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU())
            self.pos_query_layer = nn.Sequential(nn.Linear(in_channel, d_model, bias=False), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.input_layer = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.mask_layer = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        self.mask_embed_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        # attention layer
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        self.lin_squeeze = nn.ModuleList([])
        for i in range(num_layer):
            self.cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.self_attn_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
            self.lin_squeeze.append(nn.Linear(block_channels[i], d_model))

        self.lin_out_cross_attn_layers = nn.ModuleList([])
        self.lin_out_self_attn_layers = nn.ModuleList([])
        self.lin_out_ffn_layers = nn.ModuleList([])
        for _ in range(num_decode):
            self.lin_out_cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.lin_out_self_attn_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.lin_out_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))

        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_class + 1))
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

        self.atten_mask_pooling = spconv.SparseAvgPool3d(kernel_size=2, stride=2)

        for m in [self.mask_embed_head, self.mask_layer, self.out_cls, self.out_score]:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.lin_squeeze:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        for p in self.query_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if not use_feat_query:
            for p in self.pos_query_layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def get_mask(self, query, mask_feats, voxel_coords, spatial_shape, pooling_step):
        pred_mask = torch.einsum('nd,md->nm', query, mask_feats)
        attn_mask = None
        if self.attn_mask:
            attn_masks = spconv.SparseConvTensor(pred_mask.T, voxel_coords.int(), spatial_shape, 1)
            for _ in range(pooling_step):
                attn_masks = self.atten_mask_pooling(attn_masks)

            attn_mask = (attn_masks.features.T.sigmoid() < 0.5).bool()  # (num_querys, num_superpoints)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False    # True的位置不进行attend，weight会置为-inf，False的位置保持不变
            attn_mask = attn_mask.detach()

        return pred_mask, attn_mask

    def prediction(self, query, mask_feats,
                   v2p_map, sparse_batch_idxs, batch_idx, voxel_coords, spatial_shape, pooling_step):
        query = self.out_norm(query)
        mask_embed = self.mask_embed_head(query)
        pred_labels = self.out_cls(query)                                   # (num_querys, num_class+1)
        pred_scores = self.out_score(query)                                 # (num_querys, 1)
        pred_masks, attn_masks = self.get_mask(mask_embed, mask_feats,      # (num_querys, num_superpoints), (num_querys, num_superpoints)
                                               voxel_coords, spatial_shape, pooling_step)

        pred_masks_temp = torch.zeros((pred_masks.shape[0], sparse_batch_idxs.shape[0]), dtype=pred_masks.dtype, device=pred_masks.device)
        pred_masks_temp[:, torch.where(sparse_batch_idxs == batch_idx)] = pred_masks
        pred_masks = pred_masks_temp[:, v2p_map]
        batch_idxs = sparse_batch_idxs[v2p_map]
        pred_masks = pred_masks[:, torch.where(batch_idxs == batch_idx)]

        return pred_labels, pred_scores, pred_masks, attn_masks

    def forward(self, input, query,
                v2p_map, sparse_batch_idxs, batch_idx, voxel_coords, spatial_shape,
                aux_out):
        """
        input [B*M, inchannel]
        """
        prediction_labels = []
        prediction_masks = []
        prediction_scores = []
        pos_query = None
        if self.use_feat_query:
            query = self.query_layer(query)
        else:
            pos_query = query.unsqueeze(0)
            for i, layer in enumerate(self.pos_query_layer):
                pos_query = layer(pos_query.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(pos_query)    # (B, query, c)
            pos_query = pos_query[0]

            # query = torch.zeros_like(pos_query)
            query = query.unsqueeze(0)
            for i, layer in enumerate(self.query_layer):
                query = layer(query.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(query)    # (B, query, c)
            query = query[0]

        mask_feats = self.mask_layer(input)

        select_idx = torch.where(sparse_batch_idxs == batch_idx)[0]
        coord = voxel_coords[select_idx]
        coord[:, 0] = 0

        for i in range(self.num_layer):
            pred_labels, pred_scores, pred_masks, attn_masks = self.prediction(
                query, mask_feats,
                v2p_map, sparse_batch_idxs, batch_idx, coord, spatial_shape, self.num_layer - i - 1)
            prediction_labels.append(pred_labels)
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)

            inst_feats = aux_out[-(self.num_layer - i)]
            indices = inst_feats.indices[:, 0].int()
            inst_feats = inst_feats.features[torch.where(indices == batch_idx)]
            inst_feats = self.lin_squeeze[self.num_layer - i - 1](inst_feats)

            query = self.cross_attn_layers[i](inst_feats, query, attn_masks, pos_query)
            query = self.self_attn_layers[i](query, pos_query)
            query = self.ffn_layers[i](query)

        inst_feats = self.input_layer(input)
        for i in range(self.num_decode):
            pred_labels, pred_scores, pred_masks, attn_masks = self.prediction(
                query, mask_feats,
                v2p_map, sparse_batch_idxs, batch_idx, coord, spatial_shape, 0)
            prediction_labels.append(pred_labels)
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)

            query = self.lin_out_cross_attn_layers[i](inst_feats, query, attn_masks, pos_query)
            query = self.lin_out_self_attn_layers[i](query, pos_query)
            query = self.lin_out_ffn_layers[i](query)

        pred_labels, pred_scores, pred_masks, _ = self.prediction(
            query, mask_feats,
            v2p_map, sparse_batch_idxs, batch_idx, coord, spatial_shape, 0)
        prediction_labels.append(pred_labels)
        prediction_scores.append(pred_scores)
        prediction_masks.append(pred_masks)

        return dict(
            labels=prediction_labels,
            scores=prediction_scores,
            masks=prediction_masks,
        )


class query_instance_onlyFinalRepeat(nn.Module):

    def __init__(
        self,
        num_decode,
        # num_layer,
        num_class,
        in_channel,
        d_model,
        nhead,
        hidden_dim,
        dropout=0.0,
        activation_fn='relu',
        attn_mask=False,
        use_feat_query=True,
        only_last_loss=False,
        **kwargs
    ):
        super().__init__()

        self.num_decode = num_decode
        self.attn_mask = attn_mask

        self.use_feat_query = use_feat_query
        self.only_last_loss = only_last_loss
        if use_feat_query:
            self.query_layer = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        else:
            self.query_layer = nn.Sequential(nn.Linear(in_channel, d_model, bias=False), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            # self.pos_query_layer = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU())
            self.pos_query_layer = nn.Sequential(nn.Linear(in_channel, d_model, bias=False), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.input_layer = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.mask_layer = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.mask_embed_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        # attention layer
        # self.cross_attn_layers = nn.ModuleList([])
        # self.self_attn_layers = nn.ModuleList([])
        # self.ffn_layers = nn.ModuleList([])
        # self.lin_squeeze = nn.ModuleList([])
        # for i in range(num_layer):
        #     self.cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
        #     self.self_attn_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
        #     self.ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
        #     self.lin_squeeze.append(nn.Linear(block_channels[i], d_model))

        self.lin_out_cross_attn_layers = nn.ModuleList([])
        self.lin_out_self_attn_layers = nn.ModuleList([])
        self.lin_out_ffn_layers = nn.ModuleList([])
        for _ in range(num_decode):
            self.lin_out_cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.lin_out_self_attn_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.lin_out_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))

        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_class + 1))
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

        for m in [self.mask_embed_head, self.input_layer, self.mask_layer, self.out_cls, self.out_score]:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        # for m in self.lin_squeeze:
        #     for p in m.parameters():
        #         if p.dim() > 1:
        #             nn.init.xavier_uniform_(p)

        for p in self.query_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if not use_feat_query:
            for p in self.pos_query_layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def get_mask(self, query, mask_feats):
        pred_mask = torch.einsum('nd,md->nm', query, mask_feats)
        attn_mask = None
        if self.attn_mask:
            attn_mask = (pred_mask.sigmoid() < 0.5).bool()  # (num_querys, num_superpoints)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False    # True的位置不进行attend，weight会置为-inf，False的位置保持不变
            attn_mask = attn_mask.detach()

        return pred_mask, attn_mask

    def prediction(self, query, mask_feats,
                   v2p_map, sparse_batch_idxs, batch_idx):
        query = self.out_norm(query)
        mask_embed = self.mask_embed_head(query)
        pred_labels = self.out_cls(query)                                   # (num_querys, num_class+1)
        pred_scores = self.out_score(query)                                 # (num_querys, 1)
        pred_masks, attn_masks = self.get_mask(mask_embed, mask_feats)      # (num_querys, num_superpoints), (num_querys, num_superpoints)

        pred_masks_temp = torch.zeros((pred_masks.shape[0], sparse_batch_idxs.shape[0]), dtype=pred_masks.dtype, device=pred_masks.device)
        pred_masks_temp[:, torch.where(sparse_batch_idxs == batch_idx)[0]] = pred_masks
        pred_masks = pred_masks_temp[:, v2p_map]
        batch_idxs = sparse_batch_idxs[v2p_map]
        pred_masks = pred_masks[:, torch.where(batch_idxs == batch_idx)[0]]

        return pred_labels, pred_scores, pred_masks, attn_masks

    def forward(self, input, query,
                v2p_map, sparse_batch_idxs, batch_idx):
        """
        input [B*M, inchannel]
        """
        prediction_labels = []
        prediction_masks = []
        prediction_scores = []
        pos_query = None
        if self.use_feat_query:
            query = self.query_layer(query)
        else:
            pos_query = query.unsqueeze(0)
            for i, layer in enumerate(self.pos_query_layer):
                pos_query = layer(pos_query.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(pos_query)    # (B, query, c)
            pos_query = pos_query[0]

            # query = torch.zeros_like(pos_query)
            query = query.unsqueeze(0)
            for i, layer in enumerate(self.query_layer):
                query = layer(query.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(query)    # (B, query, c)
            query = query[0]

        mask_feats = self.mask_layer(input)

        # for i in range(self.num_layer):
        #     pred_labels, pred_scores, pred_masks, attn_masks = self.prediction(
        #         query, mask_feats,
        #         v2p_map, sparse_batch_idxs, batch_idx, coord, spatial_shape, self.num_layer - i - 1)
        #     prediction_labels.append(pred_labels)
        #     prediction_scores.append(pred_scores)
        #     prediction_masks.append(pred_masks)

        #     inst_feats = aux_out[-(self.num_layer - i)]
        #     indices = inst_feats.indices[:, 0].int()
        #     inst_feats = inst_feats.features[indices == batch_idx]
        #     inst_feats = self.lin_squeeze[self.num_layer - i - 1](inst_feats)

        #     query = self.cross_attn_layers[i](inst_feats, query, attn_masks, pos_query)
        #     query = self.self_attn_layers[i](query, pos_query)
        #     query = self.ffn_layers[i](query)

        inst_feats = self.input_layer(input)
        for i in range(self.num_decode):
            pred_labels, pred_scores, pred_masks, attn_masks = self.prediction(
                query, mask_feats,
                v2p_map, sparse_batch_idxs, batch_idx)
            prediction_labels.append(pred_labels)
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)

            query = self.lin_out_cross_attn_layers[i](inst_feats, query, attn_masks, pos_query)
            query = self.lin_out_self_attn_layers[i](query, pos_query)
            query = self.lin_out_ffn_layers[i](query)

        pred_labels, pred_scores, pred_masks, _ = self.prediction(
            query, mask_feats,
            v2p_map, sparse_batch_idxs, batch_idx)
        prediction_labels.append(pred_labels)
        prediction_scores.append(pred_scores)
        prediction_masks.append(pred_masks)

        return dict(
            labels=prediction_labels,
            scores=prediction_scores,
            masks=prediction_masks,
        )


query_instance = query_instance_onlyFinalRepeat
