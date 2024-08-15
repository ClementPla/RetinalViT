from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.morphology import gradient
from matplotlib.axes import Axes

from vitRet.models.superpixels.topo import get_segments_centroids


def imshow(
    img: torch.Tensor,
    segment: Optional[torch.Tensor] = None,
    border_width: int = 3,
    border_alpha: float = 1.0,
    title: Optional[str] = None,
    cmap: str = "jet_r",
    vmin: Optional[int] = None,
    vmax: Optional[int] = None,
    normalize: bool = False,
    ax: Optional[Axes] = None,
    border_cmap=None,
    show: bool = False,
    mask: Optional[torch.Tensor] = None,
    heatmap: Optional[torch.Tensor] = None,
    heatmap_alpha=0.5,
    binarize_heatmap: bool = False,
    show_segment_id: bool = False,
    threshold=0.5,
    font_size=8,
):
    """
    Display an image with optional segmentation overlay of the superpixels' borders.
    """
    img = img.cpu()
    segment = segment.cpu() if segment is not None else None
    img = img.float()
    if img.dim() == 4:
        img = img.squeeze(0)

    if img.dim() == 3:
        img = img.permute(1, 2, 0)

    if normalize:
        img = (img - img.min()) / (img.max() - img.min())

    img = img.numpy()
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    if mask is not None:
        mask = 1 - mask.cpu().float().numpy()
        if img.ndim == 3:
            mask = np.expand_dims(mask, axis=2)
            img = np.concatenate([img, mask], axis=2)
            mask = None

    alphas = mask
    ax.imshow(img, cmap=cmap, interpolation="none", vmin=vmin, vmax=vmax, alpha=alphas)

    if heatmap is not None:
        if heatmap.dim() == 4:
            heatmap = heatmap.squeeze()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = heatmap > threshold if binarize_heatmap else heatmap
        ax.imshow(
            heatmap,
            cmap=cmap,
            alpha=(heatmap > threshold) * heatmap_alpha,
            interpolation="nearest",
        )

    if segment is not None:
        if segment.dim() == 2:
            segment = segment.unsqueeze(0)

        if show_segment_id:
            x_segment, y_segment = get_segments_centroids(segment.long())
            for x, y in zip(x_segment[0], y_segment[0]):
                n = segment[0, y, x].item()
                ax.text(x.item(), y.item(), str(n), fontsize=font_size, color="white", ha="center", va="center")

        if segment.dim() == 3:
            segment = segment.unsqueeze(0)
        kernel = segment.new_ones(border_width, border_width)
        segment_border = (gradient(segment.float(), kernel).squeeze() > 0).cpu()
        alphas = (segment_border.float() * border_alpha).cpu().numpy()
        segment_border = segment_border.numpy()

        ax.imshow(segment_border, alpha=alphas, cmap=border_cmap)

    ax.set_axis_off()
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.margins(x=0)
    if title is not None:
        ax.set_title(title)
    if show:
        fig.tight_layout()
        fig.show()

    return fig
