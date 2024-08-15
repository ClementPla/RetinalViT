from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Integer
from torch_scatter import scatter

from vitRet.models.custom_ops.cluster import get_community_cluster
from vitRet.models.super_vit.losses import ClusterLoss
from vitRet.models.superpixels.graph import normalize_adjacency
from vitRet.models.superpixels.indexing import (
    consecutive_reindex_batch_of_integers_tensor,
    get_superpixels_batch_mask,
    reconstruct_spatial_map_from_segment,
)
from vitRet.models.superpixels.spatial import SpatialAdjacency

Array = torch.Tensor


class AffinityPooling(nn.Module):
    def __init__(
        self,
        dims: Optional[int] = None,
        project_dim: Optional[int] = None,
        n_heads: Optional[int] = None,
        prototypes: Optional[Float[Array, "resolution n_heads_dims"]] = None,  # noqa: F722 # type: ignore
        tau: float = 1,
        cosine_clustering: bool = True,
        graph_pool=True,
        initial_resolution=1e3,
        index=0,
        resolution_decay=0.75,
        cluster_algoritm="leiden",
        keep_self_loops: bool = True,
    ) -> None:
        super().__init__()
        assert dims is not None or prototypes is not None
        if prototypes is not None:
            print("Prototype passed as argument. Ignoring n_heads and dims.")
            self.prototypes = nn.Parameter(prototypes, requires_grad=True)
        else:
            print("Creating prototypes")
            self.prototypes = nn.Parameter(torch.randn(1, n_heads, dims), requires_grad=True)

        self.n_heads, self.dims = self.prototypes.shape[-2:]
        if project_dim is None:
            project_dim = self.dims
        self.project_dim = project_dim
        self.cosine_clustering = cosine_clustering
        self.cluster_algoritm = cluster_algoritm
        # if self.dims != self.project_dim:
        # self.K = nn.Linear(self.dims, self.project_dim)
        # self.Q = nn.Linear(self.dims, self.project_dim)
        # self.V = nn.Linear(self.dims, self.project_dim)
        # self.head = nn.Linear(self.project_dim, self.dims)

        # else:
        self.K = nn.Identity()
        self.Q = nn.Identity()
        self.V = nn.Identity()
        self.head = nn.Identity()

        self.tau = tau
        self.graph_pool = graph_pool
        loss_type = "SSA" if self.cosine_clustering else "KLDiv"
        print(f"Using {loss_type} loss")
        self.align_loss = ClusterLoss(loss_type=loss_type)
        self.index = index
        self.eps = 1e-9
        self.initial_resolution = initial_resolution
        self.resolution_decay = resolution_decay
        self.spatialAdjancency = SpatialAdjacency(keep_self_loops=keep_self_loops)

    def _compute_proximity_adjacency(
        self, segment: Integer[Array, "B H W"], normalize=True, binarize: bool = False
    ) -> torch.Tensor:
        """
        args:
            segment: tensor, shape (B, H, W), max(segment) = N, number of segments
        returns:
            adjacency: tensor, shape (B, N, N)
        """
        if segment.ndim == 4:
            segment = segment.squeeze(1)
        assert segment.ndim == 3

        adj = self.spatialAdjancency(segment)
        adj[:, 1:, 0] = 0
        adj[:, 0, 1:] = 0
        size_node = torch.diagonal(adj, dim1=-2, dim2=-1)  # This is the size of each node
        adj = torch.diagonal_scatter(adj, adj.sum(dim=2, keepdim=False) - size_node + 1, dim1=-2, dim2=-1)
        return normalize_adjacency(adj, normalize=normalize, binarize=binarize)

    @staticmethod
    def _compute_prototype_clustering(
        queries: Float[Array, "B N project_dim"],
        keys: Float[Array, "resolution n_heads project_dim"],
        cosine: bool = True,
        tau: float = 1,
        eps=1e-8,
    ) -> torch.Tensor:
        """
        args:
            queries: tensor, shape (B, N, C)
            keys: tensor, shape (resolution, K, C)

        """
        B, N, C = queries.shape
        R, K, C = keys.shape
        keys = keys.reshape(-1, C)
        if cosine:
            C = F.normalize(queries, p=2, dim=-1, eps=eps) @ F.normalize(keys, p=2, dim=-1, eps=eps).t()
            C = C.view(B, N, R, K).max(dim=2).values
            C = (C + 1) / 2
            S = C / C.sum(dim=-1, keepdim=True)

        else:
            dist = (
                torch.cdist(F.normalize(queries, p=2, dim=-1, eps=eps), F.normalize(keys, p=2, dim=-1, eps=eps), p=2)
                ** 2
            )  # Float[Array, "B N n_heads"]
            dist = dist.view(B, N, R, K).min(dim=2).values
            dist = (1 + dist / tau).pow(-(tau + 1.0) / 2.0)
            S = dist / dist.sum(dim=-1, keepdim=True)  # Float[Array, "B N n_heads"]
        return S

    def forward(self, x: Float[Array, "B N dims"], segments: Integer[Array, "B H W"]):
        segments = consecutive_reindex_batch_of_integers_tensor(segments)
        if segments.ndim == 3:
            segments = segments.unsqueeze(1)

        mask = get_superpixels_batch_mask(segments)

        resolution = self.initial_resolution * (self.resolution_decay**self.index) * x.shape[0]
        prox_adj = self._compute_proximity_adjacency(segments, binarize=True)
        prototype = self.prototypes

        queries, values = self.Q(x), self.V(x)
        keys = self.K(prototype)
        C = self._compute_prototype_clustering(queries, keys, cosine=self.cosine_clustering, tau=self.tau, eps=self.eps)
        loss = self.align_loss(C, tau=self.tau, mask_node=mask)

        # C = (C * self.tau).softmax(dim=-1)
        proto_adj = torch.matmul(C, C.permute(0, 2, 1))

        proto_adj = proto_adj - proto_adj.flatten(start_dim=1).min(dim=1).values.unsqueeze(1).unsqueeze(1)
        proto_adj = proto_adj / proto_adj.flatten(start_dim=1).max(dim=1).values.unsqueeze(1).unsqueeze(1)

        # print("Proto adj", proto_adj[0].min(), proto_adj[0].max())
        # plt.imshow(proto_adj[0].detach().cpu().numpy(), vmax=1, vmin=0)
        # plt.show()
        proto_adj = normalize_adjacency(proto_adj, normalize=True, binarize=False)
        adj = prox_adj * proto_adj
        # Assert the adjacency matrix is symmetric
        # torch.testing.assert_close(adj, adj.permute(0, 2, 1))

        adj_norm = normalize_adjacency(adj, normalize=False)
        v = adj_norm @ values  # This is pretty much the equivalent of Attn @ V in the original transformer
        x = self.head(v)

        if self.graph_pool:
            clustered_nodes = get_community_cluster(
                adj,
                mask_node=mask,
                resolution=resolution,
                fill_diagonal=True,
                per_batch=True,
                algorithm=self.cluster_algoritm,
                theta=1.0,
                max_iter=100,
            )
            if clustered_nodes.amax() > 128:
                segments = reconstruct_spatial_map_from_segment(clustered_nodes, segments)
                x = scatter(x, clustered_nodes, dim=1, reduce="mean")
        return x, segments, loss, C
