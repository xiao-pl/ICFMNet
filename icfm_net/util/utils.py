import functools
import os
import os.path as osp
from collections import OrderedDict
from math import cos, pi
from typing import Any, Dict, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import distributed as dist
from torch import nn

from .dist import get_dist_info, master_only


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, apply_dist_reduce=False):
        self.apply_dist_reduce = apply_dist_reduce
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def dist_reduce(self, val):
        rank, world_size = get_dist_info()
        if world_size == 1:
            return val
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val, device='cuda')
        dist.all_reduce(val)
        return val.item() / world_size

    def get_val(self):
        if self.apply_dist_reduce:
            return self.dist_reduce(self.val)
        else:
            return self.val

    def get_avg(self):
        if self.apply_dist_reduce:
            return self.dist_reduce(self.avg)
        else:
            return self.avg

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Epoch counts from 0 to N-1
def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * \
            (1 + cos(pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def is_multiple(num, multiple):
    return num != 0 and num % multiple == 0


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.
    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


@master_only
def checkpoint_save(epoch, model, optimizer, work_dir, save_freq=16):
    if hasattr(model, 'module'):
        model = model.module
    f = os.path.join(work_dir, f'epoch_{epoch}.pth')
    checkpoint = {
        'net': weights_to_cpu(model.state_dict()),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, f)
    if os.path.exists(f'{work_dir}/latest.pth'):
        os.remove(f'{work_dir}/latest.pth')
    os.system(f'cd {work_dir}; ln -s {osp.basename(f)} latest.pth')

    # remove previous checkpoints unless they are a power of 2 or a multiple of save_freq
    epoch = epoch - 1
    f = os.path.join(work_dir, f'epoch_{epoch}.pth')
    if os.path.isfile(f):
        if not is_multiple(epoch, save_freq) and not is_power2(epoch):
            os.remove(f)


def load_checkpoint(checkpoint, logger, model, optimizer=None, strict=False):
    if hasattr(model, 'module'):
        model = model.module
    device = torch.cuda.current_device()
    state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(device))
    src_state_dict = state_dict['net']
    target_state_dict = model.state_dict()
    skip_keys = []
    # skip mismatch size tensors in case of pretraining
    for k in src_state_dict.keys():
        if k not in target_state_dict:
            continue
        if src_state_dict[k].size() != target_state_dict[k].size():
            skip_keys.append(k)
    for k in skip_keys:
        del src_state_dict[k]
    missing_keys, unexpected_keys = model.load_state_dict(src_state_dict, strict=strict)
    if skip_keys:
        logger.info(
            f'removed keys in source state_dict due to size mismatch: {", ".join(skip_keys)}')
    if missing_keys:
        logger.info(f'missing keys in source state_dict: {", ".join(missing_keys)}')
    if unexpected_keys:
        logger.info(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}')

    # load optimizer
    if optimizer is not None:
        assert 'optimizer' in state_dict
        optimizer.load_state_dict(state_dict['optimizer'])

    if 'epoch' in state_dict:
        epoch = state_dict['epoch']
    else:
        epoch = 0
    return epoch + 1


def get_max_memory():
    mem = torch.cuda.max_memory_allocated()
    mem_mb = torch.tensor([int(mem) // (1024 * 1024)], dtype=torch.int, device='cuda')

    _, world_size = get_dist_info()
    if world_size > 1:
        dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)

    torch.cuda.reset_peak_memory_stats()

    return mem_mb.item()


def cuda_cast(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for x in args:
            if isinstance(x, torch.Tensor):
                x = x.cuda()
            new_args.append(x)
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.cuda()
            new_kwargs[k] = v
        return func(*new_args, **new_kwargs)

    return wrapper


def exists(val):
    return val is not None


def max_value(t):
    return torch.finfo(t.dtype).max     # finfo用于获取属性


def batched_index_select(values, indices, dim=1):
    '''
    values: (b, n, n) / (b, n, n, dim)
    indices: (b, n, num_neighbors)
    '''
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]      # 根据目前的dim值, 是将indices扩增与values一样的维度    (b, n, num_neighbors) -> (b, n, num_neighbors, 1)
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)    # 将上一步indices扩增的维度复制扩展   (b, n, num_neighbors, 1) -> (b, n, num_neighbors, dim)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]  # (slice(None),)*dim 等同与 (:,)*dim 后者语法报错

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


@torch.jit.script
def batch_sigmoid_bce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: (num_querys, N)
        targets: (num_inst, N)
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')

    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, (1 - targets))

    return loss / N


@torch.jit.script
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_weight):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.register_buffer('cost_weight', torch.tensor(cost_weight))

    @torch.no_grad()
    def forward(self, pred_labels, pred_masks, insts):
        '''
        pred_masks: List[Tensor] len(p2c) == B, Tensor.shape == (n, N)
        pred_labels: (B, n_q, 19)
        insts: List[Instances3D]
        '''
        indices = []
        for pred_label, pred_mask, inst in zip(pred_labels, pred_masks, insts):
            if len(inst) == 0:
                indices.append(([], []))
                continue
            pred_label = pred_label.softmax(-1)     # (num_querys, num_class+1)
            tgt_idx = inst.gt_labels                # (num_inst) 实例体对应的语义标签
            cost_class = -pred_label[:, tgt_idx]    # (num_querys, num_inst)

            tgt_mask = inst.gt_masks              # (num_inst, num_superpoints)
            cost_mask = batch_sigmoid_bce_loss(pred_mask, tgt_mask.float())     # (num_querys, num_inst)
            cost_dice = batch_dice_loss(pred_mask, tgt_mask.float())    # (num_querys, num_inst)

            C = (self.cost_weight[0] * cost_class + self.cost_weight[1] * cost_mask + self.cost_weight[2] * cost_dice)
            C = C.cpu()
            indices.append(linear_sum_assignment(C))    # tuple(2, min(num_querys, num_inst)) 在每行选择一个元素，使得每列只被选一次，且所有选中元素的权重和最小
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class Instances3D:
    """
    This class represents a list of instances in a scene.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``

       .. code-block:: python

          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    def __init__(self, num_points: int, gt_instances: np.ndarray = None, **kwargs: Any):
        """
        Args:
            num_points: number of points of the scene.
            kwargs: fields to add to this `Instances`.
        """
        self._num_points = num_points
        self._gt_instances = gt_instances
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def num_points(self) -> int:
        """
        Returns:
            int
        """
        return self._num_points

    @property
    def gt_instances(self) -> np.ndarray:
        """
        Returns:
            int
        """
        return self._gt_instances

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith('_'):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == '_fields' or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (len(self) == data_len), 'Adding a field of length {} to a Instances of length {}'.format(
                data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> 'Instances3D':
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances3D(self._num_points, self._gt_instances)
        for k, v in self._fields.items():
            if hasattr(v, 'to'):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def cuda(self, *args: Any, **kwargs: Any) -> 'Instances3D':
        ret = Instances3D(self._num_points, self._gt_instances)
        for k, v in self._fields.items():
            if hasattr(v, 'cuda'):
                v = v.cuda(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> 'Instances3D':
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError('Instances index out of range!')
            else:
                item = slice(item, None, len(self))

        ret = Instances3D(self._num_points, self._gt_instances)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError('Empty Instances does not support __len__!')

    def __iter__(self):
        raise NotImplementedError('`Instances` object is not iterable!')

    def __str__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self))
        s += 'num_points={}, '.format(self._num_points)
        s += 'fields=[{}])'.format(', '.join((f'{k}: {v}' for k, v in self._fields.items())))
        return s

    __repr__ = __str__


@torch.jit.script
def get_iou(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


@torch.jit.script
def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None, is_sigmoid=True, is_mean=True):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    if is_sigmoid:
        inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？

    if weights is not None:
        loss *= weights[:, 0]

    if is_mean:
        return loss.mean()
    else:
        return loss.sum()
