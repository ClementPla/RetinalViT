import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, to_dense_adj

def batch_index_from_segment(segment:torch.Tensor):
    """
    Return the batch index of each segment in the input tensor
    segment: tensor (type int64) of shape BxWxH
    """
    b = segment.shape[0]
    segments_per_image = (torch.amax(segment, dim=(1,2,3)) + 1).long()
    batch_index = torch.arange(0, b, device=segment.device).repeat_interleave(segments_per_image)
    return batch_index

def reindex_segments_from_batch(segments:torch.Tensor, with_padding=False):
    """
    Reindex the segments in the input tensor in increasing order, either in consecutive order (with_padding=False).
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
    segments_per_image = (torch.amax(segments, dim=(1,2,3)) + 1).long()
    # We reindex each segment in increasing order along the batch dimension
    batch_index = torch.arange(0, b, device=segments.device).repeat_interleave(segments_per_image)
    if with_padding:
        max_nodes = segments.amax() + 1
        padding  = torch.arange(0, b, device=segments.device) * max_nodes
        segments += padding.view(-1, 1, 1, 1)
    else:
        cum_sum = torch.cumsum(segments_per_image, -1)
        cum_sum = torch.roll(cum_sum, 1)
        cum_sum[0] = 0 
        segments += cum_sum.view(-1, 1, 1, 1)
    
    return segments, batch_index
     
def get_superpixels_adjacency(segments:torch.Tensor, keep_self_loops:bool=False):
    """
    Return the adjacency matrix for the superpixel segmentation map. 
    Each value of the adjacency matrix corresponds to the number of touching pixels between two segments.
    segments: tensor (type int64) of shape BxWxH
    """
    if segments.ndim == 3:
         segments.unsqueeze_(1)
    # Maximum number of segments in the batch
    n_segments = torch.amax(segments) + 1
    
    b = segments.shape[0]
    
    indexed_segments, batch_index = reindex_segments_from_batch(segments)
    
    # For each pixel, this kernel extracts its neighbours (connectivity 4)
    kernel = torch.zeros(4,1,2,2, device=segments.device, dtype=torch.float)
    kernel[0,:,0, 0] = 1
    kernel[1,:,0, 1] = 1
    kernel[2,:,0, 0] = 1
    kernel[3,:,1, 0] = 1
    neighbours = F.conv2d(indexed_segments.float(), kernel)
    
    # For each segment, we track its batch position
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

