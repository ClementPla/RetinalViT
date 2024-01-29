
from typing import Iterator, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp
from torch.nn.parameter import Parameter
from torch_scatter import scatter

from vitRet.models.stochastic_attention.tokenization import MultiScaleTokenization, Projector
from vitRet.models.stochastic_attention.vit_modules import Block
from vitRet.utils.images import last_dim_normalization, spatial_batch_normalization


class StochasticVisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        norm_layer: Union[nn.LayerNorm, nn.Identity] = nn.LayerNorm,
        init_values: Optional[float] = None,
        act_layer: Union[nn.GELU, nn.ReLU] = nn.GELU,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        in_chans: int = 3,
        scales: int = 3,
        single_cls_token: bool = True,
        kernel_size: int = 16,
        projection_stride: int = 16,
        img_size: int = 512,
        drop_rate: float = 0.0,
        max_tokens: int = 512,
        discard_ratio: float = 0.5,
        global_pool: bool = False,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.,
        padding: int = 0,
        first_embedding_dim=768,
        detach_input=False,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.blocks_depth = depth
        self.scales = scales
        self.embed_dim = embed_dim
        self.projector = Projector(
            padding=padding,
            in_chans=in_chans,
            out_chans=first_embedding_dim,
            stride=projection_stride,
            kernel_size=kernel_size,
            img_size=img_size,
        )

        self.tokenizer = MultiScaleTokenization(
            input_dim=first_embedding_dim,
            embed_dim=embed_dim,
            scales=scales,
            single_cls_token=single_cls_token,
            global_pool=global_pool,
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )
        self.embed_dim = embed_dim
        self.head_drop = nn.Dropout(drop_rate)
        self.norm = norm_layer(embed_dim)
        self.fc_norm = nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.global_pool = global_pool
        self.interpolation_mode = "nearest"
        self.detach_input = detach_input
        if isinstance(img_size, list) or isinstance(img_size, tuple):
            img_size = img_size[0]
        self.img_size = img_size
        if self.detach_input:
            for param in self.projector.parameters():
                param.requires_grad = False

    @property
    def is_compressed(self):
        return self.projector.out_chans != self.embed_dim
        
    @property
    def no_weight_decay(self):
        return ["projector.pos_embed", "tokenizer.cls_token", "tokenizer.cls_pos_embed"]
    
    def parameters(self, recurse: bool = True):
        params = []
        named_parameters = self.named_parameters(recurse=True)
        for name, param in named_parameters:
            if self.detach_input and "projector" in name:
                continue
            params.append(param)
        return params
    
    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, Parameter]]:
        named_parameters =  super().named_parameters(prefix, recurse, remove_duplicate)
        for name, param in named_parameters:
            if self.detach_input and "projector" in name:
                continue
            yield name, param
    
    def forward(self, img: torch.Tensor, segments: torch.Tensor, return_attention: bool = False):
        x = self.projector(img)
        if self.detach_input:
            x = x.detach()
            
        seqs, segments = self.tokenizer(x, segments)

        predictor_token, attention_map = self.forward_sequences(seqs, segments)
        predictor_token = self.fc_norm(predictor_token)
        predictor_token = self.head_drop(predictor_token)
        if predictor_token.ndim==3:
            predictor_token = predictor_token.mean(1)
        classification = self.head(predictor_token)
        if return_attention:
            attention_map = spatial_batch_normalization(attention_map)
            return classification, attention_map

        return classification
    
    def forward_features(self, x: torch.Tensor):
        """
        Importance score based on:
        https://arxiv.org/pdf/2111.15667.pdf
        Args:
            sequence (torch.Tensor): Tensor of shape B x N x C
        """
        for block in self.blocks:
            x, attn, value = block(x, return_attention=True, return_value=True)
        # Compute importance score
        v_i = value[:, :, 1:]  # B x H x N x d
        attn = attn[:, :, 0, 1:]  # B x H x N
        v_norm = v_i.norm(dim=-1, keepdim=False)  # B x H x N
        importance_score = attn * v_norm  # B x H x N
        importance_score = importance_score / (importance_score.sum(dim=-1, keepdim=True)+1e-7)  # B x H x N
        importance_score = importance_score.sum(dim=1)  # B x N
        return x, importance_score

    @torch.no_grad()
    def resample_sequence(self, sequence: torch.Tensor, segment: torch.Tensor, resampling_map: torch.Tensor):
        seg_distribution = scatter(resampling_map.flatten(1), segment.flatten(1))
        distribution = torch.nan_to_num(seg_distribution)
        distribution = distribution - distribution.min(1, keepdim=True)[0]

        indices = torch.multinomial(
            distribution,
            self.max_tokens,
            replacement=False,
        )

        # Get the segments indices we want based on the sampled indices
        cls_next = sequence[:, :1]
        sequence = sequence[:, 1:]

        # Extract the tokens in the sequence based on the segment indices
        sequence = torch.gather(sequence, 1, indices.unsqueeze(2).repeat_interleave(self.embed_dim, dim=2))
        sequence = torch.cat([cls_next, sequence], 1)
        return sequence, indices

    @torch.no_grad()
    def update_sampling_map(self, sampling_map, local_attention_map, sequence_length, segments, indices_updates):
        # Since the sequence has been subsampled, we need to reconstruct
        # the value from the local (subsampled) attention map.
        B = segments.shape[0]
        reconstruct = sampling_map.new_zeros(B, sequence_length)

        reconstruct = reconstruct.scatter_(1, indices_updates, local_attention_map)

        new_gam = torch.gather(reconstruct, 1, segments.flatten(1))

        # Update the global attention map
        gam = sampling_map.flatten()
        new_gam = new_gam.flatten()
        indices = new_gam > 0
        gam[indices] = new_gam[indices]
        sampling_map = gam.view(B, -1)
        return sampling_map

    def forward_sequences(self, sequences, segments):
        predictors_tokens = []
        B = sequences[0].shape[0]  # Batch size
        H, W = segments.shape[-2:]  # Image size
        global_attention_map = torch.zeros(B, H * W, dtype=sequences[0].dtype, device=sequences[0].device)

        for i, cur_sequence in enumerate(sequences):
            # Get the current segments
            cur_segment = segments[:, i]
            sequence_length = cur_sequence.shape[1] - 1
            is_sequence_too_long = (sequence_length > self.max_tokens) and (i > 0)

            # If the number of segment is higher than the max_tokens
            # Resample the sequence based on the global attention map
            if is_sequence_too_long:
                cur_sequence, indices = self.resample_sequence(cur_sequence, cur_segment, global_attention_map)
                cur_sequence.requires_grad = True

            # Regular forward pass
            y, local_attention_map = self.forward_features(cur_sequence)
            # local_attention_map = torch.softmax(local_attention_map, -1)
            local_attention_map = last_dim_normalization(local_attention_map)
            # Extracting the prediction token
            y = self.norm(y)
            predictors_tokens.append(y[:, :1])

            if not (is_sequence_too_long):
                # Take the local map as the new global map (the sequence is complete)
                global_attention_map = torch.gather(local_attention_map, 1, cur_segment.flatten(1))
            else:
                global_attention_map = self.update_sampling_map(
                    global_attention_map, local_attention_map, sequence_length, cur_segment, indices
                )

        if len(predictors_tokens) > 1:
            predictors_tokens = torch.cat(predictors_tokens, 1)
        else:
            predictors_tokens = predictors_tokens[0].squeeze(1)    
        global_attention_map = global_attention_map.view(-1, H, W)
        return predictors_tokens, global_attention_map


if __name__ == "__main__":
    img_size = 512
    model = StochasticVisionTransformer(
        num_classes=1, kernel_size=5, img_size=img_size, padding=0, projection_stride=16, global_pool=False
    )
    b_size = 4
    foo = torch.randn(b_size, 3, img_size, img_size)
    seq1 = torch.arange(0, 64).view(1, 1, 8, 8).repeat(b_size, 1, 1, 1).float()
    seq2 = torch.arange(0, 256).view(1, 1, 16, 16).repeat(b_size, 1, 1, 1).float()
    seq3 = torch.arange(0, 576).view(1, 1, 24, 24).repeat(b_size, 1, 1, 1).float()
    seq4 = torch.arange(0, 1024).view(1, 1, 32, 32).repeat(b_size, 1, 1, 1).float()
    sequences = [seq4, seq3, seq2, seq1]
    sequences = [F.interpolate(seq, size=(img_size, img_size), mode="nearest").long() for seq in sequences]
    sequences = torch.cat(sequences, 1)
    pred, attn_map = model(foo, sequences, return_attention=True)
