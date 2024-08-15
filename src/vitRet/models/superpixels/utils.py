from typing import Optional

import torch
import torch.nn.functional as F
from torch_scatter import scatter


def get_superpixels_groundtruth(
    gt_mask: torch.Tensor,
    superpixels_segment: torch.Tensor,
    roi: torch.Tensor,
    has_background: bool = False,
    threshold: Optional[float] = None,
    to_multiclass: bool = True,
):
    """
    superpixels_segment: (B, H, W)
    gt_mask: (B, C, H, W)
    """
    # We get the sum of each groundtruth overlapping each segment

    if not has_background:
        bg_gt = (1 - gt_mask.max(1, keepdim=True)[0]) * roi
        gt_mask = torch.cat([1 - roi, bg_gt, gt_mask], 1)
    if to_multiclass:
        n_classes = gt_mask.shape[1]
        gt_mask = gt_mask.argmax(1, keepdim=True)
        gt_mask = F.one_hot(gt_mask, num_classes=n_classes).squeeze(1).permute(0, 3, 1, 2)

    flatten_segment = superpixels_segment.flatten(1)
    pixels_per_class_per_segment = scatter(
        gt_mask.flatten(-2).long(), flatten_segment.unsqueeze(1), reduce="sum", dim=-1
    ).float()

    pixels_per_segment = (
        scatter(torch.ones_like(flatten_segment), flatten_segment, reduce="sum", dim=-1).unsqueeze(1).float()
    )

    ratio = pixels_per_class_per_segment / pixels_per_segment

    if threshold is not None:
        ratio = (ratio > threshold).float()

    ratio[:, 1] = 1 - ratio[:, 2:].max(1)[0]
    ratio[:, 0] = 1 - ratio[:, 1:].max(1)[0]

    ratio = torch.nan_to_num(ratio, nan=0)
    return ratio
