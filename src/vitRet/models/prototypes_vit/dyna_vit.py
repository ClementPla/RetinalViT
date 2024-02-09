import math
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from timm.models.helpers import named_apply
from timm.models.vision_transformer import Block, Mlp, PatchDropout
from torch_geometric.nn.pool.max_pool import max_pool_x
from torch_geometric.utils import dense_to_sparse, to_dense_batch
from torch_scatter import scatter

from vitRet.models.prototypes_vit.cluster.cluster_edges import cluster_index
from vitRet.models.prototypes_vit.utils import get_superpixels_adjacency, reindex_segments_from_batch


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()
            

class SegmentEmbed(nn.Module):
    def __init__(self, kernel_size, in_chans=3, inter_dim=64, embed_dim: int = 768,
                 img_size: Union[int, Tuple[int, int]] = 224,):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv1 = nn.Conv2d(in_chans, inter_dim, kernel_size=kernel_size, 
                               stride=1, padding='same')
        
        self.pos_embed = nn.Parameter(torch.randn(1, inter_dim, img_size, img_size) * 0.02 )
        self.conv2 = nn.Conv1d(inter_dim, embed_dim, kernel_size=3, padding=1)

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x, segment):
        x = self.conv1(x)
        x = x + self.pos_embed
        if segment.ndim == 3:
            segment.unsqueeze_(1)
        f_segment = segment.flatten(1)
        _, f_segment = torch.unique(f_segment, return_inverse=True)
        x = scatter(x.flatten(2), f_segment.unsqueeze(1), reduce="mean")
        x = self.conv2(x).permute(0, 2, 1)
        return x, f_segment.view(segment.shape)


class DynViT(nn.Module):
    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            inter_dim: int = 64,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,
            mlp_layer: Callable = Mlp,
            number_of_prototypes: int = 16,
            ) -> None:
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        self.num_prefix_tokens = 1 if class_token else 0

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
        self.segment_embed = SegmentEmbed(kernel_size=patch_size,
                                          in_chans=in_chans, 
                                          inter_dim=inter_dim, 
                                          embed_dim=embed_dim, 
                                          img_size=img_size)
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights()
    
    def no_weight_decay(self):
        return {'segment_embed.pos_embed', 'cls_token', 'dist_token', 'prototypes'}
    
    def init_weights(self):
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def _resample_sequence(self, x, segments, depth_index):
        
        prox_adj = self._compute_proximity_adjacency(segments)
        prototype = self.prototypes[depth_index]
        proto_adj = self._compute_prototype_adjacency(x, prototype)
        adj = prox_adj * proto_adj
        
        # Assert the adjacency matrix is symmetric
        # torch.testing.assert_close(adj, adj.permute(0, 2, 1))

        edge_index, edge_attribute = dense_to_sparse(adj)
        clustered_nodes = cluster_index(edge_index).to(segments.device)
        
        # B x N, max(clustered_nodes) = n_cluster
        segments = torch.gather(clustered_nodes, -1, segments.flatten()).view(segments.shape)
        _ , segments = torch.unique(segments, return_inverse=True) 
        _, batch_index = reindex_segments_from_batch(segments)
        Nmax = segments.max() + 1

        b, N = x.shape[:2]
        # x: B x N x C
        x = adj @ x  # This is pretty much the equivalent of Attn @ V in the original transformer  
        # x: B x N x C

        x = x.view(b*N, -1) # We flatten all the tokens along the batch dimension
        x = scatter(x, clustered_nodes.unsqueeze(1), dim=0, reduce='mean')
        # x: (B x Nmax) x C

        x, _ = to_dense_batch(x, batch_index, batch_size=b, max_num_nodes=Nmax)
        # x: B x Nmax x C

        return x, segments

        
    @staticmethod
    def normalize_adjacency(adj: torch.Tensor, normalize: bool = True, binarize = False) -> torch.Tensor:
        """
        args:
            adj: tensor, shape (B, N, N)
            normalize: bool
        returns:
            adjacency: tensor, shape (B, N, N)shape
        """
        if binarize:
            return (adj > 0).float()
        if normalize:
            degree = adj.sum(dim=2, keepdim=True)
            degree = degree.pow(-0.5)
            degree[degree == float('inf')] = 0
            
            diag = torch.eye(degree.shape[1], device=degree.device).unsqueeze(0).expand_as(adj)
            diag = diag * degree
            adj =  diag @ adj @ diag 
        return adj

    def _compute_proximity_adjacency(self, segment, normalize=True) -> torch.Tensor:
        """
        args:
            segment: tensor, shape (B, H, W), max(segment) = N, number of segments
        returns:
            adjacency: tensor, shape (B, N, N)
        """
        adj = get_superpixels_adjacency(segment, keep_self_loops=True)
        return self.normalize_adjacency(adj, normalize=normalize)
    
    def _compute_prototype_adjacency(self, x, prototypes, normalize=True) -> torch.Tensor:
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
        cosine_sim = (cosine_sim + 1) / 2
        cosine_sim[cosine_sim < 0.15] = 0
        class_adj = torch.matmul(cosine_sim, cosine_sim.permute(0, 2, 1))
        return self.normalize_adjacency(class_adj, normalize=normalize)

    def _segment_embed(self, x, segment):
        return x, segment
    
    def forward_features(self, x, segment):
        x, segment = self.segment_embed(x, segment)
        x = self.norm_pre(x)
        for d, blk in enumerate(self.blocks):
            x = blk(x)
            x, segment = self._resample_sequence(x, segment, d)
        x = self.norm(x)
        return x
    
    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

        
    def forward(self, x, segment):
        """
        args:
            x: tensor, shape (B, C, H, W)
            segment: tensor, shape (B, H, W)
        """
        
        x = self.forward_features(x, segment)
        x = self.forward_head(x)
        return x
    
    
if __name__ == '__main__':

    model = DynViT(img_size=512, patch_size=16, in_chans=3, num_classes=1000, inter_dim=64,
                   embed_dim=96).cuda(2)

    x = torch.randn(3, 3, 512, 512).cuda(2)
    segment = torch.arange(0, 16*16).reshape(1, 16, 16).repeat(3, 1, 1).cuda(2)
    segment = torch.nn.functional.interpolate(segment.float().unsqueeze(1), size=(512, 512), mode='nearest').long().squeeze(1)
    out = model(x, segment)
    print(out)