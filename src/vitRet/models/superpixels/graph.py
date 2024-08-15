import torch


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
