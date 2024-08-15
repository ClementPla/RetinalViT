import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple
from torch_scatter import scatter

from vitRet.models.superpixels.indexing import consecutive_reindex_batch_of_integers_tensor


class DinoSegmentEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        self.dino.patch_embed.proj.stride = (8, 8)
        self.dino.patch_embed.proj.padding = (3, 3)
        self.dino.patch_size = 8

    @property
    def embed_size(self):
        return 384

    def forward(self, x, segment):
        self.eval()
        if segment.ndim == 3:
            segment.unsqueeze_(1)

        img_size = to_2tuple(x.shape[-2:])
        if img_size[0] % 14 != 0:
            new_size = to_2tuple((img_size[0] // 14) * 14)
            x = F.interpolate(x, size=new_size, mode="bilinear")

        x = self.dino.forward_features(x)["x_norm_patchtokens"]
        N_tokens, f = x.shape[1:]
        h = w = int(N_tokens**0.5)
        x = x.permute(0, 2, 1).view(-1, f, h, w)
        numel = x.shape[0] * x.shape[1] * segment.shape[2] * segment.shape[3]
        if numel > 2147483647:
            splits = 2147483647 // (x.shape[1] * segment.shape[2] * segment.shape[3])
            xsplit = torch.split(x, splits, dim=0)
            x = torch.cat([F.interpolate(x_, size=segment.shape[-2:], mode="bilinear") for x_ in xsplit], 0)
        else:
            x = F.interpolate(x, size=segment.shape[-2:], mode="bilinear")
        segment = consecutive_reindex_batch_of_integers_tensor(segment)
        x = scatter(x.flatten(2), segment.flatten(2), reduce="mean").permute(0, 2, 1)

        return x, segment

    def init_weights(self):
        print("DinoSegmentEmbedding does not require initialization")
