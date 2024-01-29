import torch


def spatial_batch_normalization(batch:torch.Tensor):
    ndim = batch.ndim
    min_value = torch.amin(batch, (ndim-2, ndim-1), keepdim=True)
    max_value = torch.amax(batch, (ndim-2, ndim-1), keepdim=True)
    return (batch - min_value) / (max_value - min_value)

def last_dim_normalization(batch:torch.Tensor):
    min_value = torch.amin(batch, -1, keepdim=True)
    max_value = torch.amax(batch, -1, keepdim=True)
    return (batch - min_value) / (max_value - min_value)
