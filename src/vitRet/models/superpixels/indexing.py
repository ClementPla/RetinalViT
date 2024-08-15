from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch_geometric.utils as tg_utils
from jaxtyping import Integer
from torch_geometric.utils import remove_self_loops, to_dense_adj, unbatch
from torch_scatter import scatter

Array = torch.Tensor


def batch_index_from_segment(segment: Integer[Array, "B 1 H W"]) -> torch.Tensor:
    """
    Return the batch index of each segment in the input tensor
    segment: tensor (type int64) of shape BxWxH
    """
    b = segment.shape[0]
    segments_per_image = (torch.amax(segment, dim=(1, 2, 3)) + 1).long()
    batch_index = torch.arange(0, b, device=segment.device).repeat_interleave(segments_per_image)
    return batch_index


def reindex_batch_segments(segments: Integer[Array, "B 1 H W"]):
    return torch.cat([torch.unique(s, return_inverse=True)[1].unsqueeze(0) for s in segments], 0)


def consecutive_reindex_batch_of_integers_tensor(input_batch):
    org_shape = input_batch.shape
    input_batch = input_batch.flatten(1)

    # Calculate the maximum value for each tensor in the batch
    max_vals, _ = torch.max(input_batch, dim=1)
    offsets = max_vals.cumsum(0)
    offsets = torch.roll(offsets, 1, 0)
    offsets[0] = 0
    # Apply offsets
    offset_tensor = input_batch + offsets.unsqueeze(-1)
    # Flatten, globally reindex, and then split back
    flat_tensor = offset_tensor.flatten()
    _, global_indices = torch.unique(flat_tensor, return_inverse=True)
    output_tensor = global_indices.view(org_shape[0], -1)
    # Make sure that all indices start at 0
    min_vals, _ = torch.min(output_tensor, dim=1, keepdim=True)
    output_tensor = output_tensor - min_vals
    return output_tensor.view(org_shape)


def reindex_all_segments_across_batch(segments: torch.Tensor, with_padding=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reindex the segments in the input tensor in increasing order, in consecutive order (with_padding=False).
    If with_padding=True, the segments are reindexed in increasing order with padding between each batch element
    so that each segment on the i-th batch element is indexed by i*num_nodes + segment_index.
    segments: tensor (type int64) of shape BxWxH
    e.g: with padding False:
    segments = [[0, 1],
                [0, 3],
                [2, 3]]
    reindex_segments_from_batch(segments, with_padding=False):
    tensor([[0, 1],
            [2, 5],
            [8, 9]])
    e.g: with padding True:
    reindex_segments_from_batch(segments, with_padding=True):
    tensor([[0, 1], # Missing 2, 3 (NMax = 4)
            [4, 7], # Missing 5, 6
            [10, 11]]) # Missing 8, 9

    """
    segments = segments.clone()
    b = segments.shape[0]
    segments_per_image = (torch.amax(segments, dim=(1, 2, 3)) + 1).long()
    # We reindex each segment in increasing order along the batch dimension
    if with_padding:
        max_nodes = segments.amax() + 1
        padding = torch.arange(0, b, device=segments.device) * max_nodes
        padding[-1] = padding[-1] - 1
        segments += padding.view(-1, 1, 1, 1)
        batch_index = torch.arange(0, b, device=segments.device).repeat_interleave(segments_per_image)
    else:
        cum_sum = torch.cumsum(segments_per_image, -1)
        cum_sum = torch.roll(cum_sum, 1)
        cum_sum[0] = 0
        segments += cum_sum.view(-1, 1, 1, 1)
        batch_index = torch.cat([segments.new_ones(s) * i for i, s in enumerate(segments_per_image)], 0)

    return segments, batch_index


def get_superpixels_batch_mask(segment: Integer[Array, "B N"]) -> torch.Tensor:
    segments_max = segment.amax((1, 2, 3))
    B = segment.shape[0]
    N = segments_max.amax() + 1
    mask = torch.arange(0, N, device=segment.device).unsqueeze(0).expand(B, -1)
    mask = mask < segments_max.unsqueeze(1)
    return mask


def reconstruct_spatial_map_from_segment(x, segments, reindex=True):
    """
    Expects x of shape (B, F, N) where N is the number of segments (i.e N=segments.max()+1)
    """
    if segments.ndim == 4:
        segments = segments.squeeze(1)
    segments = segments.long()
    b, h, w = segments.shape
    if x.ndim == 2:
        x = x.unsqueeze(1)
    f = x.shape[1]
    segments = segments.unsqueeze(1).expand(-1, f, -1, -1)
    reconstructed_map = torch.gather(x.flatten(2), -1, segments.flatten(2)).view(b, f, h, w).squeeze(1)
    if reindex:
        return reindex_batch_segments(reconstructed_map)
    return reconstructed_map


def unbatch_and_stack(x: torch.Tensor, batch_index: torch.Tensor):
    """
    x: tensor (N, C)
    batch_index: tensor (N)
    """
    out_unbatch = unbatch(x, batch_index)
    return pad_and_stack(out_unbatch)


def pad_and_stack(x: torch.Tensor):
    # Works along dim 0
    max_lengh = max([_out.shape[0] for _out in x])
    out_stacked = [torch.nn.functional.pad(_out, (0, 0, 0, (max_lengh - _out.shape[0])), value=0) for _out in x]
    out_stacked = torch.stack(out_stacked, dim=0)
    return out_stacked


def pad_and_concat(x: torch.Tensor):
    # Works along dim 1
    max_lengh = max([_out.shape[1] for _out in x])
    out_stacked = [torch.nn.functional.pad(_out, (0, 0, 0, (max_lengh - _out.shape[1]))) for _out in x]
    out_stacked = torch.cat(out_stacked, dim=0)
    return out_stacked


def select_by_index(
    x: Array,
    batch: Array,
    fill_value: float = 0.0,
    max_num_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tuple[Array, Array]:
    """
    taken from torch_geometric.utils.to_dense_batch
    """

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    num_nodes = tg_utils.scatter(batch.new_ones(x.size(0)), batch, dim=0, dim_size=batch_size, reduce="sum")
    cum_nodes = tg_utils.cumsum(num_nodes)
    filter_nodes = False

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())
    elif num_nodes.max() > max_num_nodes:
        filter_nodes = True

    tmp = torch.arange(batch.size(0), device=x.device) - cum_nodes[batch]
    idx = tmp + (batch * max_num_nodes)
    if filter_nodes:
        mask = tmp < max_num_nodes
        x, idx = x[mask], idx[mask]

    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = torch.as_tensor(fill_value, device=x.device)
    out = out.to(x.dtype).repeat(size)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool, device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask
