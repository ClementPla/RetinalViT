from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, to_dense_adj

from vitRet.models.superpixels.indexing import reindex_all_segments_across_batch


class SpatialAdjacency(nn.Module):
    def __init__(self, keep_self_loops: bool = False):
        super().__init__()

        self.keep_self_loops = keep_self_loops

        kernel_x = torch.zeros(2, 1, 1, 2, dtype=torch.float)
        kernel_x[0, :, 0, 0] = 1.0
        kernel_x[1, :, 0, 1] = 1.0
        kernel_y = kernel_x.permute(0, 1, 2, 3)

        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    @torch.no_grad()
    def forward(self, segments):
        if segments.ndim == 3:
            segments = segments.unsqueeze(1)

        n_segments = torch.amax(segments) + 1

        b = segments.shape[0]

        indexed_segments, batch_index = reindex_all_segments_across_batch(segments)

        xx = F.conv2d(indexed_segments.float(), self.kernel_x).permute(1, 0, 2, 3).reshape(2, -1).long()
        yy = F.conv2d(indexed_segments.float(), self.kernel_y).permute(1, 0, 2, 3).reshape(2, -1).long()
        edge_index = torch.cat([xx, yy], 1)
        if not self.keep_self_loops:
            edge_index, _ = remove_self_loops(edge_index=edge_index)

        adj = to_dense_adj(edge_index, batch=batch_index, batch_size=b, max_num_nodes=int(n_segments))
        adj = (adj + adj.permute(0, 2, 1)) / 2

        if self.keep_self_loops:
            diag = torch.diagonal(adj, dim1=-2, dim2=-1)
            diag /= 2

        return adj
