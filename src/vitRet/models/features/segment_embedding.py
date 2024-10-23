from typing import Tuple, Union

import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from timm.layers.helpers import to_2tuple
from torch.nn import functional as F

from vitRet.models.custom_ops.superpixel_features_scatter import scatter_segments
from vitRet.models.superpixels.indexing import reindex_batch_segments


class SegmentEmbed(nn.Module):
    def __init__(
        self,
        in_chans=3,
        n_conv_stem=1,
        inter_dim: int = 128,
        embed_dim: int = 768,
        kernel_size: int = 16,
    ):
        super().__init__()
        seqs = []
        prev_channel = in_chans
        for _ in range(n_conv_stem):
            seqs.append(
                nn.Sequential(
                    nn.Conv2d(prev_channel, inter_dim, kernel_size=kernel_size, stride=1, padding="same"), nn.ReLU()
                )
            )
            prev_channel = inter_dim

        self.convs = nn.Sequential(*seqs)
        self.pos_embed = nn.Parameter(torch.randn(1, inter_dim, 64, 64) * 0.02)
        if inter_dim != embed_dim:
            self.proj = nn.Conv1d(inter_dim, embed_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()
        self._embed_size = embed_dim

    @property
    def embed_size(self):
        return self._embed_size

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, segment):
        features = self.convs(x)
        segment = reindex_batch_segments(segment)
        features = scatter_segments(features, segment, "mean", "bilinear")
        scatter_pos_embed = scatter_segments(self.pos_embed, segment, "mean", "bilinear")
        features = features + scatter_pos_embed
        features = self.proj(features)
        return features.permute(0, 2, 1), segment


def test():
    torch.autograd.set_detect_anomaly(True)
    from vitRet.data.data_factory import get_test_sample

    sample = get_test_sample(batch_size=16)

    image = sample["image"].cuda()
    segment = sample["segments"].cuda()
    model = SegmentEmbed().cuda()

    out = model(image, segment)
    l = out[0].sum()
    l.backward()
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)

    # torch.cuda.synchronize()
    print(out[0].shape)


if __name__ == "__main__":
    test()
