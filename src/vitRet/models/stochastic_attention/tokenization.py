import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


def conv2d_output_size(input_size, kernel_size, stride, padding, dilation=1):
    """Compute the output size of a convolutional layer"""
    return math.floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class Projector(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        out_chans: int = 128,
        img_size: int = 512,
        stride: int = 16,
        kernel_size: int = 16,
        padding: int = 0,
        act_layer: Union[nn.Identity, nn.GELU, nn.ReLU, nn.Sigmoid] = nn.Identity,
    ) -> None:
        super().__init__()
        if in_chans == out_chans:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding)
            
        self.proj_act = act_layer()
        self.out_chans = out_chans
        if isinstance(img_size, list) or isinstance(img_size, tuple):
            img_size = img_size[0]

        if padding == "same" and stride == 1:
            output_size = img_size
        else:
            output_size = conv2d_output_size(img_size, kernel_size, stride, padding)

        self.pos_embed = nn.Parameter(torch.randn(1, out_chans, output_size, output_size) * 0.02)

    def forward(self, x):
        """
        Projects the input image to a feature map.
        Adds the positional embedding to the feature map.
        Preserves the spatial arrangement (no flattening).
        Args:
            x: Tensor of shape B x C x H' x H' (expect square image, could easily be changed)
        Returns a tensor of shape B x C x H x W
        """
        x = self.proj(x)
        x = x + self.pos_embed
        x = self.proj_act(x)
        return x


class MultiScaleTokenization(nn.Module):
    def __init__(
        self, input_dim, embed_dim: int = 128, scales: int = 4, single_cls_token: bool = True, global_pool=False
    ) -> None:
        super().__init__()
        self.scales = scales
        self.single_cls_token = single_cls_token
        self.global_pool = global_pool
        if not self.global_pool:
            if self.single_cls_token:
                cls_tk = torch.zeros(1, input_dim)
            else:
                cls_tk = torch.zeros(1, scales, input_dim)
            self.cls_token = nn.Parameter(cls_tk)
            self.cls_pos_embed = nn.Parameter(torch.randn_like(self.cls_token) * 0.02)
        self.input_dim = input_dim
        if input_dim == embed_dim:
            self.conv = nn.Identity()
        else:
            self.conv = nn.Conv1d(input_dim, embed_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, segments: torch.Tensor):
        """
        :param x: Tensor of shape B x C x H x W. Note that C is typically not 3 but the embedding size.
        Extract tokens at different resolutions from the input tensor.
        The tokens are just superpixels of the input tensor.
        It will also add a classification token to the sequence's input.
        return: a list of tensors of shape B x N_scale x C where N_scale is the number of tokens at the scale

        """
        tokens = []
        B = x.shape[0]
        if not self.global_pool:
            cls_token = self.cls_token + self.cls_pos_embed
        segments.requires_grad = False
        if segments.shape[-2:] != x.shape[-2:]:
            segments = F.interpolate(segments.float(), size=x.shape[-2:], mode="nearest").long()
        segments = segments.permute(1, 0, 2, 3).flatten(2)
        for i, f_seg in enumerate(segments):
            _, f_seg = torch.unique(f_seg, return_inverse=True)
            segments[i] = f_seg
            sequence = scatter(x.flatten(2), f_seg.unsqueeze(1), reduce="mean")
            if not self.global_pool:
                if self.single_cls_token:
                    sequence = torch.cat([cls_token.expand(B, -1, -1).permute(0, 2, 1), sequence], 2)
                else:
                    sequence = torch.cat([cls_token[:, i].unsqueeze(1).expand(B, -1, -1).permute(0, 2, 1), sequence], 2)
            sequence = self.conv(sequence).permute(0, 2, 1)
            tokens.append(sequence)
            
            
        segments = segments.permute(1, 0, 2).reshape(B, -1, x.shape[-2], x.shape[-1])
        return tokens, segments

    def load_cls_pos_embed(self, other_cls_pos_embed):
        if self.global_pool:
            return
        if not self.single_cls_token:
            other_cls_pos_embed = torch.repeat_interleave(other_cls_pos_embed, self.cls_pos_embed.shape[1], 1)
        self.cls_pos_embed = torch.nn.Parameter(other_cls_pos_embed)

    def load_cls_token(self, other_cls_token):
        if self.global_pool:
            return
        if not self.single_cls_token:
            other_cls_token = torch.repeat_interleave(other_cls_token, self.cls_token.shape[1], 1)
        self.cls_token = torch.nn.Parameter(other_cls_token)
