from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, to_dense_adj
from torch_scatter import scatter


def batch_index_from_segment(segment: torch.Tensor):
    """
    Return the batch index of each segment in the input tensor
    segment: tensor (type int64) of shape BxWxH
    """
    b = segment.shape[0]
    segments_per_image = (torch.amax(segment, dim=(1, 2, 3)) + 1).long()
    batch_index = torch.arange(0, b, device=segment.device).repeat_interleave(segments_per_image)
    return batch_index


def reindex_batch_segments(segments: torch.Tensor):
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
    min_vals, _ = torch.min(output_tensor, dim=1, keepdim=True)
    output_tensor = output_tensor - min_vals
    return output_tensor.view(org_shape)


def reindex_all_segments_across_batch(segments: torch.Tensor, with_padding=False):
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
    batch_index = torch.arange(0, b, device=segments.device).repeat_interleave(segments_per_image)
    if with_padding:
        max_nodes = segments.amax() + 1
        padding = torch.arange(0, b, device=segments.device) * max_nodes
        padding[-1] = padding[-1] - 1
        segments += padding.view(-1, 1, 1, 1)
    else:
        cum_sum = torch.cumsum(segments_per_image, -1)
        cum_sum = torch.roll(cum_sum, 1)
        cum_sum[0] = 0
        segments += cum_sum.view(-1, 1, 1, 1)

    return segments, batch_index


def get_superpixels_adjacency(segments: torch.Tensor, keep_self_loops: bool = False, threshold: Optional[int] = None):
    """
    Return the adjacency matrix for the superpixel segmentation map.
    Each value of the adjacency matrix corresponds to the number of touching pixels between two segments.
    segments: tensor (type int64) of shape BxWxH or Bx1xWxH
    """
    if segments.ndim == 3:
        segments = segments.unsqueeze(1)
    # Maximum number of segments in the batch
    n_segments = torch.amax(segments) + 1

    b = segments.shape[0]

    indexed_segments, batch_index = reindex_all_segments_across_batch(segments)

    # For each pixel, this kernel extracts its neighbours (connectivity 4)
    kernel_x = torch.zeros(2, 1, 1, 2, device=segments.device, dtype=torch.float)
    kernel_x[0, :, 0, 0] = 1
    kernel_x[1, :, 0, 1] = 1
    kernel_y = kernel_x.permute(0, 1, 2, 3)
    xx = F.conv2d(indexed_segments.float(), kernel_x).permute(1, 0, 2, 3).reshape(2, -1).long()
    yy = F.conv2d(indexed_segments.float(), kernel_y).permute(1, 0, 2, 3).reshape(2, -1).long()
    # For each segment, we track its batch position

    edge_index = torch.cat([xx, yy], 1)
    # 2 x 2*n_edges (H-1 x W-1)

    if not keep_self_loops:
        edge_index, _ = remove_self_loops(edge_index=edge_index)

    adj = to_dense_adj(edge_index, batch=batch_index, batch_size=b, max_num_nodes=int(n_segments))
    adj = (adj + adj.permute(0, 2, 1)) / 2

    if threshold is not None:
        adj = adj * (adj > threshold)

    if keep_self_loops:
        diag = torch.diagonal(adj, dim1=-2, dim2=-1)
        diag /= 2

    return adj


def normalize_adjacency(adj: torch.Tensor, normalize: bool = True, binarize=False) -> torch.Tensor:
    """
    args:
        adj: tensor, shape (B, N, N)
        normalize: bool
    returns:
        adjacency: tensor, shape (B, N, N)shape
    """
    if binarize:
        return (adj > 0).float()
    if normalize:
        degree = adj.sum(dim=2, keepdim=False)
        degree = degree.pow(-0.5)
        degree[degree == float("inf")] = 0
        diag = torch.zeros_like(adj)
        # diag = torch.eye(degree.shape[1], device=degree.device).unsqueeze(0).expand_as(adj)
        diag = torch.diagonal_scatter(diag, degree, dim1=1, dim2=2)
        adj = diag @ adj @ diag

    return adj


def reconstruct_spatial_map_from_segment(x, segments):
    if segments.ndim == 4:
        segments = segments.squeeze(1)
    segments = segments.long()
    b, h, w = segments.shape
    if x.ndim == 2:
        x = x.unsqueeze(1)
    f = x.shape[1]
    segments = segments.unsqueeze(1).expand(-1, f, -1, -1)
    reconstructed_map = torch.gather(x.flatten(2), -1, segments.flatten(2)).view(b, f, h, w).squeeze(1)
    return consecutive_reindex_batch_of_integers_tensor(reconstructed_map)


def get_superpixels_groundtruth(gt_mask: torch.Tensor,
                                superpixels_segment: torch.Tensor,
                                roi: torch.Tensor,
                                has_background: bool = False,
                                threshold: Optional[float] = None):
    """
    superpixels_segment: (B, H, W)
    gt_mask: (B, C, H, W)
    """
    # We get the sum of each groundtruth overlapping each segment

    if not has_background:
        bg_gt = (1 - gt_mask.max(1, keepdim=True)[0]) * roi
        gt_mask = torch.cat([1 - roi, bg_gt, gt_mask], 1)
    flatten_segment = superpixels_segment.flatten(1)
    pixels_per_class_per_segment = scatter(gt_mask.flatten(-2).long(),
                                           flatten_segment.unsqueeze(1),
                                           reduce="sum",
                                           dim=-1)

    pixels_per_segment = scatter(torch.ones_like(flatten_segment),
                                 flatten_segment,
                                 reduce="sum",
                                 dim=-1).unsqueeze(1)

    ratio = pixels_per_class_per_segment / pixels_per_segment
    if threshold is not None:
        ratio = (ratio > threshold).float()

    ratio[:, 1] = 1 - ratio[:, 2:].max(1)[0]
    # ratio[:, 0] = 1 - ratio[:, 1:].max(1)[0]
    ratio = torch.nan_to_num(ratio, nan=0)
    return ratio
