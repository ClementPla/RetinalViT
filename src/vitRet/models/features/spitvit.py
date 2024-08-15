import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple
from torch_geometric.utils import to_dense_batch

from vitRet.models.custom_ops.bilinear_features_extractor.bili_color_ex import color_extractor
from vitRet.models.custom_ops.kde.gaussian import gaussian_KDE
from vitRet.models.superpixels.indexing import (
    consecutive_reindex_batch_of_integers_tensor,
    reindex_all_segments_across_batch,
    unbatch_and_stack,
)


class SPITViTFeatures(nn.Module):
    def __init__(
        self, img_size, beta=8, embed_size=None, concat_positional_embedding=True, include_texture_descriptor=True
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        self.beta = beta
        H, W = img_size
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="xy")
        pos = torch.cat([grid_x.unsqueeze(0), grid_y.unsqueeze(0)], dim=0)
        self.register_buffer("pos", pos)

        grid_x_reduced, grid_y_reduced = torch.meshgrid(
            torch.linspace(-1, 1, beta), torch.linspace(-1, 1, beta), indexing="xy"
        )
        sampling_points_pos = torch.cat([grid_x_reduced.unsqueeze(0), grid_y_reduced.unsqueeze(0)], dim=0)
        sampling_points_pos = einops.rearrange(sampling_points_pos, "c h w -> (h w) c")
        self.register_buffer("sampling_points_pos", sampling_points_pos)

        grid_x_reduced, grid_y_reduced = torch.meshgrid(
            torch.linspace(-torch.pi, torch.pi, beta), torch.linspace(0, 0.5, beta), indexing="xy"
        )
        sampling_points_gradient = torch.cat([grid_x_reduced.unsqueeze(0), grid_y_reduced.unsqueeze(0)], dim=0)
        sampling_points_gradient = einops.rearrange(sampling_points_gradient, "c h w -> (h w) c")
        self.register_buffer("sampling_points_gradient", sampling_points_gradient)

        gradient_kernel_x = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32)
        gradient_kernel_y = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32)
        gradient_kernel = (
            torch.stack([gradient_kernel_x, gradient_kernel_y], dim=0).view(1, 2, 3, 3).permute(1, 0, 2, 3)
        )
        self.register_buffer("gradient_kernel", gradient_kernel / (0.5 * torch.sum(torch.abs(gradient_kernel))))
        self.register_buffer("sampling_points_color", torch.linspace(0, 1.0, self.beta))

        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing="ij")
        color_sampling_points = torch.cat([grid_x.unsqueeze(0), grid_y.unsqueeze(0)], dim=0).unsqueeze(0)
        self.register_buffer("color_sampling_points", color_sampling_points)

        f_size = 3 * beta**2
        if include_texture_descriptor:
            f_size += beta**2
        if concat_positional_embedding:
            f_size += beta**2

        self.include_texture_descriptor = include_texture_descriptor
        self.concat_positional_embedding = concat_positional_embedding

        if embed_size is None:
            self.linear = nn.Identity()
            self.pos_linear = nn.Identity()
            self._embed_size = f_size
        else:
            if not concat_positional_embedding:
                self.pos_linear = nn.Linear(beta**2, embed_size)
            else:
                self.pos_linear = nn.Identity()

            self.linear = nn.Linear(f_size, embed_size)
            self._embed_size = embed_size

    @property
    def embed_size(self):
        return self._embed_size

    def position_features(self, stacked_segment, batch_size, batch_index, max_node, argsort):
        pos = self.pos.unsqueeze(0).expand(batch_size, -1, -1, -1)
        pos = einops.rearrange(pos, "b c h w -> (b h w) c")
        pos = pos[argsort]
        out_pos, mask = to_dense_batch(pos, stacked_segment, fill_value=torch.nan, max_num_nodes=max_node)
        out = self.gaussian_kde_nd(out_pos, self.sampling_points_pos, mask_nan=mask, h_coeff=0.1)
        out_unbatch = unbatch_and_stack(out, batch_index)
        return out_unbatch

    def color_features(self, img, non_stacked_segment, stacked_segment, batch_index, max_node, argsort):
        B = img.shape[0]
        color_sp = self.color_sampling_points.expand(B, -1, -1, -1)
        color_sp = einops.rearrange(color_sp, "b c h w -> (b h w) c")
        color_sp = color_sp[argsort]
        out_pos, mask = to_dense_batch(color_sp.float(), stacked_segment, fill_value=torch.nan, max_num_nodes=max_node)
        min_pos = torch.nan_to_num(out_pos, nan=torch.inf)
        max_pos = torch.nan_to_num(out_pos, nan=-torch.inf)
        bounding_box = torch.cat(
            [
                min_pos[:, :, 0].amin(dim=1, keepdim=True),
                min_pos[:, :, 1].amin(dim=1, keepdim=True),
                max_pos[:, :, 0].amax(dim=1, keepdim=True),
                max_pos[:, :, 1].amax(dim=1, keepdim=True),
            ],
            dim=1,
        )
        bounding_box = unbatch_and_stack(bounding_box, batch_index)
        result = color_extractor(
            img, non_stacked_segment.squeeze(1), bounding_box.long(), beta=self.beta, isolateSuperpixel=False
        )
        return result.flatten(-3)

    def texture_features(self, img, stacked_segment, batch_index, max_node, argsort):
        img_avg = img.mean(dim=1, keepdim=True)
        img_avg = (img_avg - img_avg.amin(dim=(1, 2, 3), keepdim=True)) / (
            img_avg.amax(dim=(1, 2, 3), keepdim=True) - img_avg.amin(dim=(1, 2, 3), keepdim=True)
        )
        gradients = F.conv2d(img_avg, self.gradient_kernel, padding=1)
        x = einops.rearrange(gradients, "b c h w -> (b h w) c")
        x = x[argsort]
        out_gradient, mask = to_dense_batch(x, stacked_segment, fill_value=torch.nan, max_num_nodes=max_node)
        mag_gradient = torch.norm(out_gradient, dim=-1, p=2, keepdim=True)
        angle_gradient = torch.atan2(out_gradient[:, :, 1], out_gradient[:, :, 0])
        out_gradient = torch.cat([angle_gradient.unsqueeze(-1), mag_gradient], dim=-1)
        out = self.gaussian_kde_nd(out_gradient, self.sampling_points_gradient, mask_nan=mask, h_coeff=0.5)
        out_unbatch = unbatch_and_stack(out, batch_index)
        return out_unbatch

    @staticmethod
    def gaussian_kde_nd(points, sampling_points, mask_nan, h_coeff=0.01):
        return gaussian_KDE(points, sampling_points, mask_nan, h_coeff)

    @staticmethod
    def gaussian_kde_1d(points, sampling_points, mask_nan, h_coeff=0.1):
        return gaussian_KDE(points.unsqueeze(-1).contiguous(), sampling_points.unsqueeze(-1), mask_nan, h_coeff)

    def forward_features(self, x, segment):
        B, _, H, W = x.shape
        seg, batch_index = reindex_all_segments_across_batch(segment, with_padding=False)
        _, counts = torch.unique(seg, return_counts=True)

        # Maximum number of pixels in a segment (excepted the black areas, which are the B-largest ones by construction)
        # (True only for fundus images with large non-roi borders)
        max_node = torch.sort(counts, descending=True).values[B]

        stacked_segment = einops.rearrange(seg.squeeze(1), "b h w -> (b h w)")
        arg_sort = torch.argsort(stacked_segment)
        stacked_segment = stacked_segment[arg_sort]
        features = []
        col_features = self.color_features(x, segment, stacked_segment, batch_index, max_node, argsort=arg_sort)
        features.append(col_features)
        pos_features = self.position_features(stacked_segment, B, batch_index, max_node, arg_sort)
        pos_features = self.pos_linear(pos_features)
        if self.concat_positional_embedding:
            features.append(pos_features)

        if self.include_texture_descriptor:
            texture_features = self.texture_features(x, stacked_segment, batch_index, max_node, arg_sort)
            features.append(texture_features)

        if len(features) > 1:
            features = torch.cat(features, dim=-1)
        else:
            features = features[0]
        features = torch.nan_to_num(features, nan=x.amin())
        features = self.linear(features)
        if not self.concat_positional_embedding:
            features += pos_features
        return features

    def forward(self, x: torch.Tensor, segment: torch.Tensor, *args, **kwargs):
        segment = consecutive_reindex_batch_of_integers_tensor(segment)
        features = self.forward_features(x, segment)
        return features, segment
