from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, to_dense_adj
from torch_scatter import scatter_add, scatter_mean

from vitRet.models.superpixels.indexing import reindex_all_segments_across_batch


@torch.no_grad()
def filter_small_segments(segments: torch.Tensor, min_size=0):
    """
    Filter small segments from the superpixels' segmentation mask by assigning them to the index 0.
    """
    B, H, W = segments.shape
    segment_size = count_segment_occurence(segments)

    mask = torch.gather(segment_size > min_size, 1, segments.flatten(1))
    mask = mask.view(B, H, W)
    segments = torch.where(mask, segments, torch.zeros_like(segments)).to(torch.int64)
    return segments


@torch.no_grad()
def count_segment_occurence(segments: torch.Tensor):
    """
    Count the number of occurrences of each segment in the superpixels' segmentation mask.
    """

    ones = torch.ones_like(segments)

    segment_sum = scatter_add(ones.flatten(1), segments.flatten(1), dim=1)

    return segment_sum


@torch.no_grad()
def get_segments_centroids(segments: torch.Tensor):
    B, H, W = segments.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=segments.device), torch.arange(W, device=segments.device))

    xx = xx.unsqueeze(0).expand(B, -1, -1)
    yy = yy.unsqueeze(0).expand(B, -1, -1)
    x_segment = scatter_mean(xx.flatten(1), segments.flatten(1), dim=1)
    y_segment = scatter_mean(yy.flatten(1), segments.flatten(1), dim=1)

    return x_segment, y_segment


@torch.no_grad()
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
    kernel_x[0, :, 0, 0] = 1.0
    kernel_x[1, :, 0, 1] = 1.0
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
