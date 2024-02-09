from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, Mlp

from vitRet.models.prototypes_vit.utils import get_superpixels_adjacency


class DynViT(nn.Module):
    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,
            mlp_layer: Callable = Mlp,
            number_of_prototypes: int = 12,
            ) -> None:
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
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
            for i in range(depth)])
        
        
        self.prototypes = nn.Parameter(torch.randn(depth, number_of_prototypes, embed_dim))
        
        
    def forward_features(self, x, segment):
        
        for d, blk in enumerate(self.blocks):
            x = blk(x)
            x, segment = self._resample_sequence(x, segment, d)
        return x

    def _resample_sequence(self, x, segment, depth_index):
        
        prox_adj = self._compute_proximity_adjacency(segment)
        protoype = self.prototypes[depth_index]
        proto_adj = self._compute_prototype_adjacency(x, protoype)
        
        
    
    def _compute_proximity_adjacency(self, segment) -> torch.Tensor:
        """
        args:
            segment: tensor, shape (B, H, W), max(segment) = N, number of segments
        returns:
            adjacency: tensor, shape (B, N, N)
        """
        return get_superpixels_adjacency(segment, keep_self_loops=True)
    
    def _compute_prototype_adjacency(self, x, prototypes) -> torch.Tensor:
        """
        args:
            x: tensor, shape (B, N, C)
            prototypes: tensor, shape (K, C)
        returns:
            adjacency: tensor, shape (B, N, N)
        """
        x_norm = F.normalize(x, dim=2)
        protos_norm = F.normalize(prototypes, dim=1)
        cosine_sim = x_norm @ protos_norm.t()
        cosine_sim[cosine_sim < 0.5] = 0
        class_adj = torch.matmul(cosine_sim, cosine_sim.permute(0, 2, 1))

        return class_adj
         
    def _segment_embed(self, x, segment):
        return x, segment
        
        
    def forward(self, x, segment):
        """
        args:
            x: tensor, shape (B, C, H, W)
            segment: tensor, shape (B, H, W)
        """
        x, segment = self._segment_embed(x=x, segment=segment)
        
        
        x = self.forward_features(x, segment)
        return x
    
if __name__ == '__main__':
    
    x = torch.randn(3, 56, 128)
    protos = torch.randn(12, 128)
    x_norm = F.normalize(x, dim=2)
    protos_norm = F.normalize(protos, dim=1)
    cosine_sim = x_norm @ protos_norm.t()
    
    torch.testing.assert_close(cosine_sim[2], F.cosine_similarity(protos.unsqueeze(2), x[2].t().unsqueeze(0), dim=1).t())
    
    cosine_sim = (cosine_sim + 1) / 2 # Squeeze to [0, 1]
    print(cosine_sim.shape, cosine_sim.min(), cosine_sim.max())
    class_adj = torch.matmul(cosine_sim, cosine_sim.permute(0, 2, 1))
    
    torch.testing.assert_close(class_adj[0],  cosine_sim[0] @ cosine_sim[0].t())
    print(class_adj.shape, class_adj.min(), class_adj.max())
    
    
    