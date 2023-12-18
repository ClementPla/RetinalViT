import math
from typing import Any, Optional, Union

import kornia as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, Mlp


def conv2d_output_size(input_size, kernel_size, stride, padding, dilation=1):
    """Compute the output size of a convolutional layer"""
    return math.floor(
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


class Projector(nn.Module):
    def __init__(
        self,
        in_chans: int=3,
        out_chans: int=128,
        img_size: int=512,
        stride: int=16,
        kernel_size: int=16,
        act_layer: Union[nn.Identity, nn.GELU, nn.ReLU, nn.Sigmoid]=nn.Identity,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        # padding = 0
        self.proj = nn.Conv2d(
            in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.proj_act = act_layer()

        if isinstance(img_size, list):
            img_size = img_size[0]

        output_size = conv2d_output_size(img_size, kernel_size, stride, padding)
        self.n_tokens = output_size**2

        self.pos_embed = nn.Parameter(
            torch.randn(1, out_chans, output_size, output_size) * 0.02
        )

    def forward(self, x):
        """
        Projects the input image to a feature map. 
        Adds the positional embedding to the feature map. 
        Preserves the spatial arrangement (no flattening).
        Returns a tensor of shape B x C x H x W
        """
        x = self.proj(x)
        x = x + self.pos_embed
        x = self.proj_act(x)
        return x


class MultiScaleTokenization(nn.Module):
    def __init__(self, embed_dim:int=128, scales:int=4, single_cls_token:bool=True) -> None:
        super().__init__()
        self.scales = scales
        self.single_cls_token = single_cls_token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1 if self.single_cls_token else scales, embed_dim)
        )

        self.cls_pos_embed = nn.Parameter(
            torch.randn(1, 1 if self.single_cls_token else scales, embed_dim) * 0.02
        )

    def forward(self, x:torch.Tensor):
        """
        :param x: Tensor of shape B x C x H x W. Note that C is typically note 3 but whatever the projector outputs.
        Extract tokens at different resolutions from the input tensor. 
        The input tensor is downsampled by a factor of 2 at each scale.
        The tokens are just patches of the input tensor.
        It will also add a classification token to the sequence's input.
        return: a list of tensors of shape B x N_scale x C where N_scale is the number of tokens at the scale 
        
        """
        tokens = []
        B = x.shape[0]
        cls_token = self.cls_token + self.cls_pos_embed
        for i in range(self.scales):
            curr = x[:, :, :: 2**i, :: 2**i]
            sequence = curr.flatten(2, 3).permute(0, 2, 1)  # B x N x C
            if self.single_cls_token:
                sequence = torch.cat([cls_token.expand(B, -1, -1), sequence], 1)
            else:
                sequence = torch.cat(
                    [cls_token[:, i].unsqueeze(1).expand(B, -1, -1), sequence], 1
                )

            tokens.append(sequence)
        return tokens

    def load_cls_pos_embed(self, other_cls_pos_embed):
        if not self.single_cls_token:
            other_cls_pos_embed = torch.repeat_interleave(
                other_cls_pos_embed, self.cls_pos_embed.shape[1], 1
            )
        self.cls_pos_embed = torch.nn.Parameter(other_cls_pos_embed)

    def load_cls_token(self, other_cls_token):
        if not self.single_cls_token:
            other_cls_token = torch.repeat_interleave(
                other_cls_token, self.cls_token.shape[1], 1
            )
        self.cls_token = torch.nn.Parameter(other_cls_token)


class Attention(nn.Module):
    def __init__(
        self,
        dim:int,
        num_heads:int=8,
        qkv_bias: bool=False,
        qk_norm: bool=False,
        norm_layer: Union[nn.LayerNorm, nn.Identity] = nn.LayerNorm,
        proj_drop: float=0.0,
        attn_drop: float=0.0,
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor, return_attention:bool=True):
        """
        Returns the output of the attention layer and the attention matrix
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # q, k, v: B x num_heads x N x head_dim
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn_presoftmax = q @ k.transpose(-2, -1)  # B x num_heads x N x N
        attn_predrop = attn_presoftmax.softmax(dim=-1)
        attn = self.attn_drop(attn_predrop)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attention:
            return x, attn_predrop
        return x


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values:float =1e-5, inplace: bool=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim:int,
        num_heads:int=8,
        mlp_ratio:float=4.0,
        qkv_bias:bool=False,
        qk_norm:bool=False,
        norm_layer: Union[nn.LayerNorm, nn.Identity]=nn.LayerNorm,
        init_values:Optional[float]=None,
        act_layer: Union[nn.GELU, nn.ReLU]=nn.GELU,
        proj_drop: float=0.0,
        attn_drop: float=0.0,
        drop_path: float=0.0,
    ) -> None:
        """Act like the Vision Transformer Block except that it can also returns the
        local Attention matrix.
        Args:
            dim (int): Number of feature per token
            num_heads (int, optional): number of heads. Defaults to 8.
            mlp_ratio (float, optional): Size of MLP relative to dim. Defaults to 4..
            qkv_bias (bool, optional): Use bias in QKV projection. Defaults to True.
            qk_norm (bool, optional): Normalized QKV projection. Defaults to True.
            norm_layer (nn.Module, optional): Normalize the output of the Attention module (V). Defaults to nn.LayerNorm.
            init_values (float, optional): Layer Scaling. Defaults to None.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
            proj_drop (float, optional): Post-projection Dropout. Defaults to 0..
            attn_drop (float, optional): Post-Attention Dropout. Defaults to 0..
            drop_path (float, optional): Final layer Path Dropout. Defaults to 0..
        """

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
        )

        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, return_attention=True):
        if return_attention:
            x1, attn = self.attn(self.norm1(x), return_attention)
        else:
            x1 = self.attn(self.norm1(x), return_attention)

        x = x + self.drop_path1(self.ls1(x1))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        if return_attention:
            return x, attn
        return x


class Scale(nn.Module):
    def __init__(
        self,
        embed_dim:int =768,
        depth:int =4,
        num_heads:int =12,
        mlp_ratio:float =4.0,
        qkv_bias: bool=False,
        qk_norm: bool=False,
        norm_layer: Union[nn.LayerNorm, nn.Identity]=nn.LayerNorm,
        init_values: Optional[float]=None,
        act_layer: Union[nn.GELU, nn.ReLU]=nn.GELU,
        proj_drop:float=0.0,
        drop_path:float=0.0,
        max_tokens:int=512,
        discard_ratio:float=0.9,
    ) -> None:
        """
        A Scale is a succession of Block (similar to a small Transformer encoder).
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

    def forward(
        self, curr_seq, next_seq=None, curr_seq_indices=None, sampling_map=None
    ):
        """

        Args:
            sequence1 (torch.Tensor): Tensor of shape B x N1 x C
            sequence2 (torch.Tensor): Tensor of shape B x N2 x C (with N2>N1)
            sampling_map (torch.Tensor): Tensor of shape B x Nmax x Nmax
        """

        # Forward pass thourgh blocks traversal and computation of attention rollout
        curr_seq, attn_rollout = self.forward_pass_with_attn_rollout(curr_seq)

        # Check if the current sequence is the last one
        is_final_block = next_seq is None

        # Use the attention rollout to update the sampling map
        expected_size = None

        # By default the next sequence indices is non existent
        next_seq_indices = None

        if not is_final_block:
            expected_size = int((next_seq.shape[1] - 1) ** 0.5)
            next_sampling_map = self.update_sampling_map(
                attn_rollout, sampling_map, curr_seq_indices, expected_size
            )

            # Resample the next sequence based on the updated sampling map
            next_seq, next_seq_indices = self.stochastic_conditional_sampling(
                next_seq, next_sampling_map
            )
        else:
            next_sampling_map = self.get_cls_attn(attn_rollout)

        return curr_seq, next_seq, next_seq_indices, next_sampling_map

    def forward_pass_with_attn_rollout(self, sequence):
        """Compute the forward pass through the blocks and the attention rollout

        Args:
            sequence (torch.Tensor): Tensor of shape B x N x C

        Returns:
            (torch.Tensor, Torch.Tensor): Tensor of shape B x N x C,
            Tensor of shape B x N x N (sequence, attention rollout)
        """
        B = sequence.shape[0]
        k = len(self.blocks)
        k = 3
        for i, block in enumerate(self.blocks):
            sequence, attn = block(sequence, return_attention=True)
            attn = attn.detach()
            attn = attn.mean(axis=1)  # Average along the heads B x N x N
            if i == len(self.blocks) - k:
                result = attn
            elif i > len(self.blocks) - k:
                a = 0.5 * (attn + torch.eye(attn.shape[1], device=attn.device))
                a = a / a.sum(dim=-1, keepdim=True)
                result = torch.matmul(a, result)

        # Checking the last map gave better results than attention rollout
        # result = attn.view(B, -1)
        attn_rollout = result.view(B, sequence.shape[1], sequence.shape[1])

        return sequence, attn_rollout  # B x N x C, B x N x N

    def get_cls_attn(self, attn_rollout):
        cls_attn = attn_rollout[:, 0, 1:]  # B x N1
        return cls_attn

    def update_sampling_map(
        self, attn_rollout, sampling_map, curr_seq_indices=None, expected_size=None
    ):
        """Update the sampling map based on the attention rollout.
        The global sampling map is updated by averaging the local attention map
        at the indices of the current sequence.

        Args:
            attn_rollout (torch.Tensor): Tensor of shape B x N1 x N1
            sampling_map (torch.Tensor): Tensor of shape B x N x N (with N>=N1)
            curr_seq_indices (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        cls_attn = self.get_cls_attn(attn_rollout)
        if sampling_map is None:
            sampling_map = cls_attn
        elif curr_seq_indices is None:
            # Corresponds to the layer where there is no stochastic sampling:
            # sequences.size(1) < max_tokens
            sampling_map = cls_attn
        else:
            B = sampling_map.shape[0]
            N1 = sampling_map.shape[1]
            # new_map = torch.zeros_like(sampling_map)

            sampling_map.scatter_reduce_(
                dim=1,
                index=curr_seq_indices,
                src=cls_attn,
                reduce="mean",
                include_self=True,
            )

            # new_map = new_map.view(B, 1, int(N1**0.5), int(N1**0.5))
            # # new_map = K.filters.gaussian_blur2d(
            # #     new_map, (11, 11), (0.5, 0.5), border_type="constant", separable=False
            # # )
            # import cv2
            # struct_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # new_map = K.morphology.dilation(new_map, kernel=torch.from_numpy(struct_el).to(new_map))
            # new_map = new_map.view(B, N1)
            # # new_map = F.normalize(new_map, dim=1, p=2)
            # mask = new_map > 0
            # ema = 0.1
            # sampling_map = mask * ((1 - ema) * new_map + ema * sampling_map) + (
            #     ~mask
            # ) * (sampling_map)
            

        # sampling_map = F.normalize(sampling_map, p=2, dim=1)
        B = sampling_map.shape[0]
        N1 = sampling_map.shape[1]

        sampling_grid = sampling_map.reshape(B, 1, int(N1**0.5), int(N1**0.5))
        if expected_size:
            sampling_grid = F.interpolate(
                sampling_grid, (expected_size, expected_size), mode="bilinear"
            )
        else:
            # If size is unspecified, assume the progression follows power of two
            sampling_grid = F.interpolate(
                sampling_grid, scale_factor=2, mode="bilinear"
            )
        sampling_map = sampling_grid.reshape(B, -1)
        return sampling_map

    def stochastic_conditional_sampling(self, sequence, sampling_map):
        """

        Args:
            sequence (torch.Tensor): Tensor of shape B x N2 x C
            sampling_map (torch.Tensor): Tensor of shape B x N2-1

        Returns:
            torch.Tensor: Tensor of shape B x N' x C
        """

        cls_token = sequence[:, 0, :].unsqueeze(1)
        sequence = sequence[:, 1:, :]
        _, N2, C = sequence.shape
        assert (
            sampling_map.shape[1] == N2
        ), f"Sampling map and current sequence should have the same length but got {sampling_map.shape[1]} and {N2}"
        # Sampling
        if N2 > self.max_tokens:
            distribution = torch.nan_to_num(sampling_map)
            distribution = torch.clamp(sampling_map, 0.5, 1.0)
            indices = torch.multinomial(
                distribution, self.max_tokens, replacement=False
            )
            indices_repeated = indices.unsqueeze(2).repeat(1, 1, C)
            sequence = torch.gather(sequence, 1, indices_repeated)
        else:
            indices = None
        sequence = torch.cat([cls_token, sequence], 1)
        return sequence, indices


class StochasticVisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes:int=1000,
        embed_dim:int=768,
        depth:int=12,
        num_heads:int=12,
        mlp_ratio:float=4.0,
        qkv_bias:bool=False,
        qk_norm:bool=False,
        norm_layer: Union[nn.LayerNorm, nn.Identity]=nn.LayerNorm,
        init_values: Optional[float]=None,
        act_layer=nn.GELU,
        proj_drop:float=0.0,
        drop_path:float=0.0,
        in_chans:int=3,
        scales:int=3,
        single_cls_token:bool=True,
        kernel_size:int=16,
        projection_stride:int=16,
        img_size:int=512,
        drop_rate: float = 0.0,
        max_tokens:int=512,
        discard_ratio:float=0.5,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.blocks_depth = depth
        self.scales = scales
        self.projector = Projector(
            in_chans=in_chans,
            out_chans=embed_dim,
            stride=projection_stride,
            kernel_size=kernel_size,
            img_size=img_size,
        )

        initial_tokens = self.projector.n_tokens
        for i in range(scales):
            print(
                f"At scale {self.scales -i }, the sequence has approx. {initial_tokens//(2**(2*i))} tokens"
            )

        self.tokenizer = MultiScaleTokenization(
            embed_dim=embed_dim, scales=scales, single_cls_token=single_cls_token
        )
        self.blocks = Scale(
            embed_dim,
            depth=self.blocks_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            act_layer=act_layer,
            proj_drop=proj_drop,
            init_values=init_values,
            max_tokens=max_tokens,
            drop_path=drop_path,
            discard_ratio=discard_ratio,
        )
        self.head_drop = nn.Dropout(drop_rate)
        self.fc_norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    @property
    def no_weight_decay(self):
        return ["projector.pos_embed", "tokenizer.cls_token", "tokenizer.cls_pos_embed"]

    def forward(self, img, return_attention=False):
        x = self.projector(img)
        seqs = self.tokenizer(x)[::-1]  # From the smallest to the longest sequence

        if self.scales > 1:
            cls_token, attention_map = self.multi_scale_forward(seqs)
        else:
            cls_token, attention_map = self.single_scale_forward(seqs[0])

        cls_token = self.fc_norm(cls_token)
        classification = self.head(cls_token)
        if return_attention:
            rows = int(attention_map.shape[1] ** 0.5)
            attention_map -= attention_map.min(1, keepdim=True)[0]
            attention_map /= attention_map.max(1, keepdim=True)[0]

            attention_map = attention_map.reshape(
                -1, rows, rows
            )  # Reshape the Global-Scale Attention Map

            # attention_map = F.normalize(attention_map, p=2, dim=(1, 2))
            return classification, attention_map

        return classification

    def multi_scale_forward(self, scales):
        attention_map = None
        indices = None
        cls_tokens = []
        scurr = scales[0]
        snext = scales[1]
        for i in range(1, self.scales):
            scurr, snext, indices, attention_map = self.blocks(
                scurr, snext, indices, attention_map
            )
            cls_tokens.append(
                scurr[:, 0].unsqueeze(1)
            )  # Store the classification token

            scurr = snext
            snext = scales[i + 1] if i + 1 < len(scales) else None

        cls_tokens = torch.cat(cls_tokens, 1).mean(1)
        return cls_tokens, attention_map

    def single_scale_forward(self, scale):
        output, _, _, attention_map = self.blocks(scale, None, None, None)
        cls_token = output[:, 0]
        return cls_token, attention_map

