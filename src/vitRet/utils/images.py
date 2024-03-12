import torch


def spatial_batch_normalization(batch:torch.Tensor):
    min_value = torch.amin(batch, (-2, -1), keepdim=True)
    max_value = torch.amax(batch, (-2, -1), keepdim=True)
    return (batch - min_value) / (max_value - min_value)

def last_dim_normalization(batch:torch.Tensor):
    min_value = torch.amin(batch, -1, keepdim=True)
    max_value = torch.amax(batch, -1, keepdim=True)
    return (batch - min_value) / (max_value - min_value)
