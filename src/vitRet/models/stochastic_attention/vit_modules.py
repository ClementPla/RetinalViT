from typing import Optional, Union

import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp
from timm.models.vision_transformer import Attention as timmAttn
from timm.models.vision_transformer import Block as timmBlock


class Attention(timmAttn):
    
    def forward(self, x: torch.Tensor, return_attention: bool = True, return_value: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn_predrop = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_predrop)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        output = (x, )
        if return_attention:
            output += (attn_predrop, )
        if return_value:
            output += (v, )
        return output

class Block(timmBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = Attention(
            kwargs['dim'],
            num_heads=kwargs['num_heads'],
            qkv_bias=kwargs['qkv_bias'],
            qk_norm=kwargs['qk_norm'],
            attn_drop=kwargs['attn_drop'],
            proj_drop=kwargs['proj_drop'],
            norm_layer=kwargs['norm_layer'],
        )
        
    def forward(self, x: torch.Tensor, return_attention=True, return_value=False) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), return_attention, return_value)
        if not isinstance(attn_out, tuple):
            attn_out = (attn_out, )
            
        x = x + self.drop_path1(self.ls1(attn_out[0]))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return (x, *attn_out[1:])