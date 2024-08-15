from typing import Tuple, Union

import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from timm.layers.helpers import to_2tuple
from torch_scatter import scatter

from vitRet.models.superpixels.indexing import consecutive_reindex_batch_of_integers_tensor


class SegmentEmbed(nn.Module):
    def __init__(
        self,
        in_chans=3,
        n_conv_stem=4,
        inter_dim: int = 128,
        embed_dim: int = 768,
        img_size: Union[int, Tuple[int, int]] = 1024,
    ):
        super().__init__()
        seqs = []
        prev_channel = in_chans
        for _ in range(n_conv_stem):
            seqs.append(
                nn.Sequential(nn.Conv2d(prev_channel, inter_dim, kernel_size=3, stride=1, padding=1), nn.ReLU())
            )
            prev_channel = inter_dim

        self.convs = nn.Sequential(*seqs)
        img_size = to_2tuple(img_size)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, *img_size) * 0.02)
        self.proj = nn.Conv1d(inter_dim, embed_dim, kernel_size=1)
        self._embed_size = embed_dim

    @property
    def embed_size(self):
        return self._embed_size

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, segment):
        x = self.convs(x)
        x = x + self.pos_embed
        if segment.ndim == 3:
            segment.unsqueeze_(1)
        segment = consecutive_reindex_batch_of_integers_tensor(segment)
        # f_segment = segment
        x = scatter(x.flatten(2), segment.flatten(2), reduce="mean")
        x = self.proj(x)
        return x.permute(0, 2, 1), segment
