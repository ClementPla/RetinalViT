import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, to_dense_adj


def get_superpixels_adjacency(segments:torch.Tensor, keep_self_loops:bool=False):
    """
    Return the adjacency matrix for the superpixel segmentation map. 
    Each value of the adjacency matrix corresponds to the number neighboring of pixels 
    segments: tensor (type int64) of shape BxWxH
    """
    # Maximum number of segments in the batch
    n_segments = torch.amax(segments) + 1
    
    b = segments.shape[0]
    
    segments_per_image = (torch.amax(segments, dim=(1,2,3)) + 1).long()
    
    cum_sum = torch.cumsum(segments_per_image, -1)
    cum_sum = torch.roll(cum_sum, 1)
    cum_sum[0] = 0 
    
    # We reindex each segment in increasing order along the batch dimension
    indexed_segments = segments + cum_sum.view(-1, 1, 1, 1)
    
    # For each pixel, this kernel extracts its neighbours (connectivity 4)
    kernel = torch.zeros(4,1,2,2, device=segments.device, dtype=torch.float)
    kernel[0,:,0, 0] = 1
    kernel[1,:,0, 1] = 1
    kernel[2,:,0, 0] = 1
    kernel[3,:,1, 0] = 1
    neighbours = F.conv2d(indexed_segments.float(), kernel)
    
    # For each segment, we track its batch position
    batch_index = torch.arange(0, b).repeat_interleave(segments_per_image)
            
    
    edge_index_horizontal = neighbours.permute(1,0,2,3)[:2].reshape(2, -1).long()
    edge_index_vertical = neighbours.permute(1,0,2,3)[2:].reshape(2, -1).long()
    
    edge_index = torch.cat([edge_index_horizontal, edge_index_vertical], 1)
    # 2 x 2*n_edges (H-1 x W-1)
    
    if not keep_self_loops:
            edge_index, _ = remove_self_loops(edge_index=edge_index)
    
    adj = to_dense_adj(edge_index, 
                    batch=batch_index, 
                    batch_size=b, 
                    max_num_nodes=int(n_segments))
    adj = (adj + adj.permute(0,2,1)) / 2
    
    if keep_self_loops:
        diag = torch.diagonal(adj, dim1=-2, dim2=-1)
        diag /=  2
    
    return adj

