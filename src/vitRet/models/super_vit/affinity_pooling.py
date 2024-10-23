from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Integer
from torch_kmeans import KMeans
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
        prototypes: Optional = None,  # noqa: F722 # type: ignore
        tau: float = 1,
        cosine_clustering: bool = True,
        graph_pool=True,
        initial_resolution=1e3,
        index=0,
        resolution_decay=0.75,
        cluster_algoritm="leiden",
        keep_self_loops: bool = True,
        use_kmeans=False,
    ) -> None:
        super().__init__()
        assert dims is not None or prototypes is not None

        if prototypes is not None:
            self.prototypes = nn.Parameter(prototypes, requires_grad=True)
        else:
            self.prototypes = nn.Parameter(torch.randn(1, n_heads, dims), requires_grad=True)

        self.n_heads, self.dims = self.prototypes.shape[-2:]
        if project_dim is None:
            project_dim = self.dims
        self.project_dim = project_dim
        self.cosine_clustering = cosine_clustering
        self.cluster_algoritm = cluster_algoritm

        # self.K = nn.Linear(self.dims, project_dim)
        # self.Q = nn.Linear(self.dims, project_dim)
        # self.V = nn.Linear(self.dims, project_dim)
        # self.head = nn.Linear(self.dims, self.dims)

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
        self.eps = 1e-8
        self.initial_resolution = initial_resolution
        self.resolution_decay = resolution_decay
        self.spatialAdjancency = SpatialAdjacency(keep_self_loops=keep_self_loops)
        self.use_kmeans = use_kmeans
        # if self.use_kmeans:
        #     self.kmean = KMeans(n_clusters=self.n_heads, max_iter=100, verbose=False, num_init=1)

        self.update_kmeans = False

    def init_weights(self):
        return

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
        batch: Float[Array, "B N project_dim"],
        centroids: Float[Array, "n_heads project_dim"],
        cosine: bool = False,
        tau: float = 1,
        eps=1e-7,
    ) -> torch.Tensor:
        """
        args:
            queries: tensor, shape (B, N, C)
            keys: tensor, shape (K, C)

        """
        B, N, C = batch.shape
        centroids = centroids.squeeze(0)
        R, C = centroids.shape

        if cosine:
            C = F.normalize(batch, p=2, dim=-1, eps=eps) @ F.normalize(centroids, p=2, dim=-1, eps=eps).t()
            C = C.view(B, N, R)
            C = (C + 1) / 2
            S = C / C.sum(dim=-1, keepdim=True)

        else:
            dist = torch.cdist(batch, centroids, p=2)  # Float[Array, "B N n_heads"]
            dist = dist.view(B, N, R)
            numerator = 1 / (1 + dist / tau)
            power = (tau + 1.0) / 2.0
            numerator = numerator**power
            denominator = numerator.sum(dim=-1, keepdim=True)
            S = numerator / (denominator + eps)
        # S = torch.nan_to_num(S, nan=0.0)
        return S

    def forward(self, x: Float[Array, "B N F"], segments: Integer[Array, "B H W"]):
        segments = consecutive_reindex_batch_of_integers_tensor(segments)
        if segments.ndim == 3:
            segments = segments.unsqueeze(1)

        mask = get_superpixels_batch_mask(segments)

        resolution = self.initial_resolution * (self.resolution_decay**self.index) * x.shape[0]
        prox_adj = self._compute_proximity_adjacency(segments, binarize=True, normalize=False)
        queries, values = self.Q(x), self.V(x)

        self.kmeans_update(queries)
        keys = self.K(self.prototypes)
        C = self._compute_prototype_clustering(queries, keys, cosine=self.cosine_clustering, tau=1, eps=self.eps)
        loss = self.align_loss(C, tau=1, mask_node=mask)

        # C = torch.nan_to_num(C, nan=1e-7)
        C = (C * self.tau).softmax(dim=-1)  # Each superpixels is associated with a prototype

        C_norm = torch.norm(C, p=2, dim=-1, keepdim=False)
        norm_matrix = C_norm.unsqueeze(2) * C_norm.unsqueeze(1)
        proto_adj = C @ C.permute(0, 2, 1)  #

        proto_adj = proto_adj / (norm_matrix + self.eps)  # Cosine similarity

        adj = prox_adj * proto_adj
        adj = normalize_adjacency(adj, normalize=True, binarize=False)
        # plt.imshow(adj[0].cpu().numpy(), vmin=adj[0].amin(), vmax=adj[0].amax())
        # plt.show()
        # Assert the adjacency matrix is symmetric
        # torch.testing.assert_close(adj, adj.permute(0, 2, 1))

        # adj = normalize_adjacency(adj, normalize=True)
        # adj = (adj - adj.amin((1, 2), keepdim=True)) / (
        #     adj.amax((1, 2), keepdim=True) - adj.amin((1, 2), keepdim=True) + 1e-8
        # )
        # adj = normalize_adjacency(adj, normalize=True, binarize=False)
        x = adj @ values  # This is pretty much the equivalent of Attn @ V in the original transformer
        # x = self.head(v)

        if self.graph_pool:
            clustered_nodes = get_community_cluster(
                adj,
                mask_node=mask,
                resolution=resolution,
                fill_diagonal=True,
                per_batch=True,
                algorithm=self.cluster_algoritm,
                theta=1.0,
                max_iter=50,
            )
            if clustered_nodes.amax() > 128:
                segments = reconstruct_spatial_map_from_segment(clustered_nodes, segments)
                x = scatter(x, clustered_nodes, dim=1, reduce="mean")
        return x, segments, loss, C

    @torch.no_grad()
    def kmeans_update(self, features: Float[Array, "B N F"]):
        return
        if (not self.update_kmeans) or (not self.use_kmeans):
            return
        B, N, F = features.shape
        x = features.reshape(1, B * N, F)
        cluster_kmean = self.kmean(x, centers=self.prototypes.view(1, -1, F))
        alpha = 0.9
        centers = self.prototypes * (1 - alpha) + alpha * cluster_kmean.centers.view(1, self.n_heads, F)
        self.prototypes.copy_(centers)
