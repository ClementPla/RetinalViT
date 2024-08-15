from typing import Optional

import torch
import torch.nn as nn
from jaxtyping import Float, Integer

Array = torch.Tensor


class ClusterLoss(nn.Module):
    def __init__(self, loss_type="SSA", gamma=1) -> None:
        super().__init__()

        assert loss_type in ["KLDiv", "SSA"]
        self.loss_type = loss_type
        self.gamma = gamma
        self.eps = 1e-9
        if self.loss_type == "KLDiv":
            self.align_loss = nn.KLDivLoss(reduction="none", log_target=True)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(
        self, C: Float[Array, "B N K"], tau: Optional[float] = 1, mask_node: Optional[Integer[Array, "B N"]] = None
    ):
        """
        Args:
            C: Correspondance matrix, tensor, shape (B, N, K)
        """
        if mask_node.ndim == 2:
            mask_node = mask_node.unsqueeze(2)
        if self.loss_type == "KLDiv":
            P = self.logsoftmax(C * tau)
            loss_elt = self.align_loss(C.log(), P)

            loss = (loss_elt * mask_node).sum() / C.shape[0]
            # C_2 = C**2
            # P = C_2 / (C.sum(dim=1, keepdim=True) + self.eps)
            # denom = P.sum(dim=2, keepdim=True)
            # denom[C.sum(dim=2, keepdim=True) == 0.0] = 1.0
            # P = P / denom
            # P = torch.nan_to_num(P, nan=0.0)
        else:
            D = 1 - C
            if self.gamma == 0:
                D = torch.min(D, dim=1)[0]
            else:
                D = -self.gamma * torch.log(torch.sum(torch.exp(-D / self.gamma), dim=-1) + self.eps)
                mask_node.squeeze_(-1)
                D = D * mask_node
                N = mask_node.sum(dim=1, keepdim=True)
                D = D / N
            loss = torch.mean(torch.sum(D, dim=1))
        return loss
