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
            self.align_loss = nn.KLDivLoss(reduction="none", log_target=False)
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
            P = C**2
            f = (C * mask_node).sum(dim=(0, 1), keepdim=True)
            target_dist = P / f
            target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True))

            target_dist = target_dist * mask_node
            target_dist = target_dist.detach()
            loss_elt = self.align_loss(C.log(), target_dist) * mask_node
            loss = loss_elt.sum() / C.shape[0]
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
