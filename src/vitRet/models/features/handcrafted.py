import torch
from torch import nn
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter, scatter_std

from vitRet.models.superpixels.indexing import consecutive_reindex_batch_of_integers_tensor, get_superpixels_batch_mask


class HandcraftedFeatures(MessagePassing):
    def __init__(self, radius: int = 0.1, out_dim=256, eps=1e-6):
        super().__init__(aggr="add")
        self.radius = radius
        self.embed_size = out_dim
        self.linear = nn.Linear(18, out_dim)
        self.eps = eps

    def message(self, x_i, x_j, pos_i, pos_j):
        return (x_i - x_j) / (torch.norm(pos_i - pos_j, dim=-1, keepdim=True) + self.eps)

    @property
    def embed_size(self):
        return self.embed_size

    @torch.no_grad()
    def forward_features(self, x, segment):
        B, C, H, W = x.shape
        segment = consecutive_reindex_batch_of_integers_tensor(segment)
        mask = get_superpixels_batch_mask(segment)
        mean_rgb_pixels = x.mean(dim=(2, 3)).unsqueeze(-1)
        mean_rgb_tokens = scatter(x.flatten(2), segment.flatten(2), reduce="mean")  # 3
        min_rgb_tokens = scatter(x.flatten(2), segment.flatten(2), reduce="min")  # 3
        max_rgb_tokens = scatter(x.flatten(2), segment.flatten(2), reduce="max")  # 3
        std_rgb_tokens = scatter_std(x.flatten(2), segment.flatten(2))
        size_tokens = scatter(x.new_ones(B, 1, H * W, dtype=torch.int), segment.flatten(2), reduce="sum")  # 1
        N = size_tokens.shape[-1]
        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device=x.device) / W, torch.arange(H, device=x.device) / H, indexing="xy"
        )

        pos = torch.cat([grid_x.unsqueeze(0), grid_y.unsqueeze(0)], dim=0)
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1)
        center = scatter(pos.flatten(2), segment.flatten(2), dim=-1, reduce="mean")  # 2

        batch = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, N).flatten()
        center_dense = center.permute(1, 0, 2).flatten(1).permute(1, 0)
        edge_index = radius_graph(center_dense, r=self.radius, batch=batch, loop=False)

        rgb = mean_rgb_tokens.permute(1, 0, 2).flatten(1).permute(1, 0)
        C = self.propagate(edge_index=edge_index, x=rgb, pos=center_dense)
        C = C.view(B, N, -1).permute(0, 2, 1)
        C = (mean_rgb_tokens / (size_tokens * mean_rgb_pixels)) * C
        # spatial_dispersion = scatter_std(pos.flatten(2), segment.flatten(2))

        size_tokens = size_tokens / size_tokens.amax(dim=-1, keepdim=True)
        features = torch.cat(
            [
                mean_rgb_tokens,
                min_rgb_tokens,
                max_rgb_tokens,
                std_rgb_tokens,
                size_tokens,
                center,
                C,
            ],
            dim=1,
        )
        features = torch.nan_to_num(features, nan=0)
        features = features * mask.unsqueeze(1)
        features = features.permute(0, 2, 1)
        return features, segment

    def forward(self, x: torch.Tensor, segment: torch.Tensor):
        features, segment = self.forward_features(x, segment)
        features = self.linear(features)
        return features, segment
        # radius_graph
