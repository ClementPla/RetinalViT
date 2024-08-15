from functools import partial
from typing import Callable, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from timm.models.helpers import named_apply
from timm.models.vision_transformer import Block, Mlp, VisionTransformer

from vitRet.models.features.builder import FeaturesExtractor
from vitRet.models.super_vit.affinity_pooling import AffinityPooling
from vitRet.models.superpixels.indexing import (
    reconstruct_spatial_map_from_segment,
)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


class DynViT(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        num_classes: int = 1000,
        global_pool: str = "token",
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        fc_norm: Optional[bool] = None,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        inter_dim: int = 64,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Optional[Callable] = None,
        act_layer: Optional[Callable] = None,
        block_fn: Callable = Block,
        mlp_layer: Callable = Mlp,
        number_of_prototypes: int = 14,
        resample_every_n_blocks: int = 2,
        graph_pool: bool = True,
        resolution_decay: float = 0.6,
        load_prototypes: Optional[bool] = True,
        initial_resolution: float = 1e3,
        use_cosine_similarity: bool = False,
        embedder_requires_grad: bool = False,
        cluster_algorithm: str = "leiden",
        embedder: Optional[str] = "dino",
        n_conv_stem: int = 4,
        in_chans: int = 3,
        f_index: int = 4,
        embed_dim: int = 768,
        downsample_segments: bool = False,
        output_stride: int = 16,
        block_size: Optional[Tuple[int, int]] = None,
        block_stride: Optional[Tuple[int, int]] = None,
        r: float = 0.1,
        beta: int = 8,
        pretrained: Union[bool, str] = False,
        include_texture_descriptor: bool = False,
        concat_positional_embedding: bool = False,
        hub_model="ClementP/FundusDRGrading-mobilenetv3_small_100",
        minimum_segment_size: int = 0,
    ) -> None:
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"

        self.segment_embed = FeaturesExtractor(
            model_type=embedder,
            f_index=f_index,
            output_stride=output_stride,
            block_size=block_size,
            block_stride=block_stride,
            inter_dim=inter_dim,
            downsample_segments=downsample_segments,
            radius=r,
            beta=beta,
            in_chans=in_chans,
            hub_model=hub_model,
            n_conv_stem=n_conv_stem,
            embed_dim=embed_dim,
            img_size=img_size,
            include_texture_descriptor=include_texture_descriptor,
            concat_positional_embedding=concat_positional_embedding,
            minimum_segment_size=minimum_segment_size,
        )
        embed_dim = self.segment_embed.embed_size
        self.segment_embed.requires_grad_(embedder_requires_grad)
        n_pooling = depth // resample_every_n_blocks
        self.resample_every_n_blocks = resample_every_n_blocks

        if resample_every_n_blocks > 0:
            self.setup_pooling_layers(
                n_pooling=n_pooling,
                number_of_prototypes=number_of_prototypes,
                embed_dim=embed_dim,
                load_prototypes=load_prototypes,
                initial_resolution=initial_resolution,
                resolution_decay=resolution_decay,
                use_cosine_similarity=use_cosine_similarity,
                graph_pool=graph_pool,
                cluster_algorithm=cluster_algorithm,
            )

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_prefix_tokens = 1 if class_token else 0

        if not pretrained:
            self.setup_not_pretrained(
                embed_dim,
                num_classes,
                global_pool,
                depth,
                num_heads,
                mlp_ratio,
                qkv_bias,
                qk_norm,
                init_values,
                class_token,
                fc_norm,
                drop_rate,
                pos_drop_rate,
                proj_drop_rate,
                attn_drop_rate,
                drop_path_rate,
                norm_layer,
                act_layer,
                block_fn,
                mlp_layer,
            )
        else:
            model: VisionTransformer = timm.create_model(
                model_name=pretrained, pretrained=True, num_classes=num_classes
            )
            self.blocks = model.blocks
            self.pos_drop = model.pos_drop
            self.norm = model.norm
            self.fc_norm = model.fc_norm
            self.head_drop = model.head_drop
            self.head = model.head
            self.cls_token = model.cls_token

    def setup_pooling_layers(
        self,
        n_pooling,
        number_of_prototypes,
        embed_dim,
        load_prototypes,
        initial_resolution,
        resolution_decay,
        use_cosine_similarity,
        graph_pool,
        cluster_algorithm,
    ):
        if load_prototypes:
            print("Loading prototypes")
            prototypes = torch.load("checkpoints/prototypes/prototypes.ckpt", map_location="cpu").permute(0, 2, 1)
            # (number_of_prototypes, embed_dim)
            print("Prototypes shape:", prototypes.shape)
        else:
            prototypes = torch.randn(1, number_of_prototypes, embed_dim)
            # torch.nn.init.xavier_normal_(prototypes)

        self.pooling_layers = nn.ModuleList(
            [
                AffinityPooling(
                    index=i,
                    initial_resolution=initial_resolution,
                    resolution_decay=resolution_decay,
                    cosine_clustering=use_cosine_similarity,
                    prototypes=prototypes.clone(),  # We clone to prevent using the same prototypes for all layers
                    graph_pool=graph_pool,
                    cluster_algoritm=cluster_algorithm,
                )
                for i in range(n_pooling)
            ]
        )

    def setup_not_pretrained(
        self,
        embed_dim,
        num_classes,
        global_pool,
        depth,
        num_heads,
        mlp_ratio,
        qkv_bias,
        qk_norm,
        init_values,
        class_token,
        fc_norm,
        drop_rate,
        pos_drop_rate,
        proj_drop_rate,
        attn_drop_rate,
        drop_path_rate,
        norm_layer,
        act_layer,
        block_fn,
        mlp_layer,
    ):
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                block_fn(
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

        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights()

    @property
    def no_weight_decay(self):
        return {"segment_embed.pos_embed", "cls_token", "dist_token", "prototypes"}

    def init_weights(self):
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def _add_class_token(self, x: torch.Tensor) -> torch.Tensor:
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        return x

    def _separate_cls_token(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cls_token is not None:
            cls_tokens = x[:, :1]
            x = x[:, 1:]
            return x, cls_tokens
        else:
            return x, None

    def _merge_cls_token(self, x: torch.Tensor, cls_tokens: torch.Tensor) -> torch.Tensor:
        if self.cls_token is not None:
            x = torch.cat((cls_tokens, x), dim=1)
        return x

    def forward_features(self, x, segment, compute_attribution=False, return_all_segments=False):
        b, c, h, w = x.shape
        x, segment = self.segment_embed(x, segment)
        x = self._add_class_token(x)
        pool_index = 0
        global_align_cost = 0.0
        attrs = []
        segments = []
        for d, blk in enumerate(self.blocks):
            if (d % self.resample_every_n_blocks) == 0 and self.resample_every_n_blocks > 0:
                x, cls_token = self._separate_cls_token(x)
                segments.append(segment)
                x, new_segment, align_cost, align_mat = self.pooling_layers[pool_index](x, segment)
                if compute_attribution:
                    attr = self.compute_attribution(segment, align_mat)
                    attrs.append(attr)
                segment = new_segment
                pool_index += 1
                global_align_cost = align_cost + global_align_cost
                x = self._merge_cls_token(x, cls_token)

            x = blk(x)

        if self.resample_every_n_blocks < 0:
            attrs = [torch.zeros_like(segment.squeeze(), dtype=torch.uint8)] * d
            global_align_cost = x.new_zeros(1)

        x = self.norm(x)
        if segment.ndim == 3:
            segment.unsqueeze_(1)

        if return_all_segments:
            for i, seg in enumerate(segments):
                if seg.ndim == 3:
                    seg.unsqueeze_(1)
                if seg.shape[-2:] != (h, w):
                    segments[i] = F.interpolate(seg.float().unsqueeze(1), size=(h, w), mode="nearest").long().squeeze(1)

        if segment.shape[-2:] != (h, w):
            segment = F.interpolate(segment.float(), size=(h, w), mode="nearest").long()
        output = [x, segment, global_align_cost / (max(pool_index, 1))]
        if compute_attribution:
            attrs = [
                F.interpolate(a.unsqueeze(1).float(), size=(h, w), mode="nearest").long().squeeze(1) for a in attrs
            ]
            output.append(attrs)

        if return_all_segments:
            output.append(segments)
        return tuple(output)

    def compute_attribution(self, segment, align_mat):
        scores = align_mat.argmax(dim=2)
        score = reconstruct_spatial_map_from_segment(scores, segment, False)
        return score

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens :].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, segment, return_attention=False, return_all_segments=False):
        """
        args:
            x: tensor, shape (B, C, H, W)
            segment: tensor, shape (B, H, W)
        """
        x, segment, align_cost, *attrs = self.forward_features(x, segment, return_attention, return_all_segments)
        x = self.forward_head(x)
        if return_attention:
            return x, segment, align_cost, *attrs
        else:
            return x, segment, align_cost


if __name__ == "__main__":
    model = DynViT(
        img_size=1024,
        in_chans=3,
        num_classes=1000,
        resample_every_n_blocks=4,
        depth=12,
        load_prototypes=False,
        embed_dim=176,
        num_heads=8,
        embedder="cnn",
        pretrained="vit_base_patch16_384",
    ).cuda(0)

    # x = torch.randn(3, 3, 1024, 1024).cuda(0)
    # segment = torch.arange(0, 32*32).reshape(1, 32, 32).repeat(3, 1, 1).cuda(0)
    # segment = torch.nn.functional.interpolate(segment.float().unsqueeze(1), size=(1024, 1024), mode='nearest').long().squeeze(1)
    # out, cost, attr = model(x, segment, True)
    # print(attr.shape)
