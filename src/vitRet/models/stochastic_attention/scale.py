from typing import Optional, Union

import torch
import torch.nn as nn

from vitRet.models.stochastic_attention.vit_modules import Block


class Scale(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        norm_layer: Union[nn.LayerNorm, nn.Identity] = nn.LayerNorm,
        init_values: Optional[float] = None,
        act_layer: Union[nn.GELU, nn.ReLU] = nn.GELU,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        max_tokens: int = 512,
        discard_ratio: float = 0.999,
        global_pool: bool = False,
    ) -> None:
        """
        A Scale is a succession of Block (similar to a Transformer encoder).
        It passes forward its input sequence at a given scale, compute
        the Local-scale attention associated to it,
        update the Global-scale Attention
        and resamples the next sequence based on the new Global scale
        attention

        Args:
            embed_dim (int, optional): _description_. Defaults to 768.
            depth (int, optional): _description_. Defaults to 4.
            num_heads (int, optional): _description_. Defaults to 12.
            mlp_ratio (_type_, optional): _description_. Defaults to 4..
            qkv_bias (bool, optional): _description_. Defaults to True.
            qk_norm (bool, optional): _description_. Defaults to True.
            norm_layer (_type_, optional): _description_. Defaults to nn.LayerNorm.
            init_values (_type_, optional): _description_. Defaults to None.
            act_layer (_type_, optional): _description_. Defaults to nn.GELU.
            proj_drop (_type_, optional): _description_. Defaults to 0..
            drop_path (_type_, optional): _description_. Defaults to 0..
            max_tokens (int, optional): Maximum number of tokens kept after resampling. Defaults to 512.
        """
        super().__init__()
        self.global_pool = global_pool
        self.discard_ratio = discard_ratio
        self.max_tokens = max_tokens
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    norm_layer=norm_layer,
                    init_values=init_values,
                    act_layer=act_layer,
                    proj_drop=proj_drop,
                    drop_path=drop_path,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        """

        Args:
        """

        # Forward pass thourgh blocks traversal and computation of attention rollout
        if self.global_pool:
            for block in self.blocks:
                x = block(x, return_attention=False)
            return x
        else:
            x, attn_map = self.forward_with_importance_score(x)
            return x, attn_map
        

    def forward_with_importance_score(self, sequence):
        """
        Importance score based on:
        https://arxiv.org/pdf/2111.15667.pdf
        Args:
            sequence (torch.Tensor): Tensor of shape B x N x C
        """
        for block in self.blocks:
            sequence, attn, value = block(sequence, return_attention=True, return_value=True)
        # Compute importance score
        v_i = value[:, :, 1:]  # B x H x N x d
        attn = attn[:, :, 0, 1:]  # B x H x N
        v_norm = v_i.norm(dim=-1, keepdim=False)  # B x H x N
        importance_score = attn * v_norm  # B x H x N
        importance_score = importance_score / (importance_score.sum(dim=-1, keepdim=True)+1e-7)  # B x H x N
        importance_score = importance_score.sum(dim=1)  # B x N
        return sequence, importance_score
        
    def forward_pass_with_attn_rollout(self, sequence):
        """Compute the forward pass through the blocks and the attention rollout

        Args:
            sequence (torch.Tensor): Tensor of shape B x N x C

        Returns:
            (torch.Tensor, Torch.Tensor): Tensor of shape B x N x C,
            Tensor of shape B x N x N (sequence, attention rollout)
        """
        B = sequence.shape[0]
        result = None
        for i, block in enumerate(self.blocks):
            sequence, attn = block(sequence, return_attention=True)
            if i==0:
                continue
            if result is None:
                result = torch.eye(attn.size(-1), device=attn.device, dtype=attn.dtype)
            with torch.no_grad():
                attention_heads_fused = attn.max(axis=1)[0]
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1)*self.discard_ratio), -1, False)
                indices = indices[indices != 0]
                flat[0, indices] = 0
                I = torch.eye(attention_heads_fused.size(-1), device=attn.device, dtype=attn.dtype)
                a = (attention_heads_fused + 1.0*I)/2
                a = a / (a.sum(dim=-1, keepdim=True)+1e-7)
                result = torch.matmul(a, result)
                
        attn_rollout = result.view(B, sequence.shape[1], sequence.shape[1])
        attn_rollout = self.get_cls_attn(attn_rollout)
        return sequence, attn_rollout  # B x N x C, B x N x N

    def get_cls_attn(self, attn_rollout):
        cls_attn = attn_rollout[:, 0, 1:]  # B x N1
        return cls_attn

if __name__ == "__main__":
    pass
