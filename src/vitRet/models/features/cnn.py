import einops
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter

# from vitRet.models.custom_ops.bilinear_features_extractor.bili_color_ex import color_extractor
from vitRet.models.custom_ops.superpixel_features_scatter.segment_scatter import scatter_segments
from vitRet.models.features.extractor.cnn import get_feature_extractor
from vitRet.models.superpixels.indexing import (
    consecutive_reindex_batch_of_integers_tensor,
    reindex_all_segments_across_batch,
    reindex_batch_segments,
    unbatch_and_stack,
)

# @torch.no_grad()
# def extract_patch_around_superpixels(image, segment, beta, isolate_superpixel=False):
#     if segment.ndim == 3:
#         segment.unsqueeze_(1)
#     segment = consecutive_reindex_batch_of_integers_tensor(segment)

#     segment.squeeze_(1)
#     B, H, W = segment.shape

#     grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing="ij")

#     color_sampling_points = torch.cat([grid_x.unsqueeze(0), grid_y.unsqueeze(0)], dim=0).unsqueeze(0).cuda().int()
#     seg, batch_index = reindex_all_segments_across_batch(segment.unsqueeze(1), with_padding=False)
#     _, counts = torch.unique(seg, return_counts=True)
#     max_node = torch.sort(counts, descending=True).values[B]
#     stacked_segment = einops.rearrange(seg.squeeze(1), "b h w -> (b h w)")
#     arg_sort = torch.argsort(stacked_segment)
#     stacked_segment = stacked_segment[arg_sort]
#     color_sp = color_sampling_points.expand(B, -1, -1, -1)
#     color_sp = einops.rearrange(color_sp, "b c h w -> (b h w) c")
#     color_sp = color_sp[arg_sort]
#     out_pos, mask = to_dense_batch(color_sp.float(), stacked_segment, fill_value=torch.nan, max_num_nodes=max_node)
#     min_pos = torch.nan_to_num(out_pos, nan=torch.inf)
#     max_pos = torch.nan_to_num(out_pos, nan=-torch.inf)
#     bounding_box = torch.cat(
#         [
#             min_pos[:, :, 0].amin(dim=1, keepdim=True),
#             min_pos[:, :, 1].amin(dim=1, keepdim=True),
#             max_pos[:, :, 0].amax(dim=1, keepdim=True),
#             max_pos[:, :, 1].amax(dim=1, keepdim=True),
#         ],
#         dim=1,
#     )
#     bounding_box = unbatch_and_stack(bounding_box, batch_index)
#     fcolor = color_extractor(
#         image, segment.squeeze(1), bounding_box.long(), beta=beta, isolateSuperpixel=isolate_superpixel
#     )
#     if segment.ndim == 3:
#         segment.unsqueeze_(1)
#     return fcolor, segment


class BaseCNN(nn.Module):
    def __init__(self, hub_model, feature_index=4, output_features=None, cat_input=False, optimized=True):
        super().__init__()
        self.model = get_feature_extractor(hub_model, out_indices=feature_index)
        self.f_index = feature_index
        self.embed_size = self.model.feature_info[feature_index]["num_chs"]
        self.optimized = optimized

        print(f"Embed size: {self.embed_size} before projection")

        pruned_indices = [i for i in range(0, feature_index + 1)]
        self.cat_input = cat_input
        offset = 3 if cat_input else 0
        # kept_indices = self.model.prune_intermediate_layers(pruned_indices, prune_head=True)
        self.act_fn = nn.Identity()
        if output_features is not None:
            if output_features != (self.embed_size + offset):
                self.last_conv = nn.Conv1d(self.embed_size + offset, output_features, 1)
                self.embed_size = output_features
            else:
                self.last_conv = nn.Identity()
        else:
            if offset > 0:
                self.last_conv = nn.Conv1d(self.embed_size + offset, self.embed_size, 1)
            else:
                self.last_conv = nn.Identity()
            # self.embed_size += 3


# class SuperPatchCNN(BaseCNN):
#     def __init__(self, hub_model, feature_index=4, beta=8, isolate_superpixels=False, output_features=None):
#         super().__init__(hub_model, feature_index, beta, isolate_superpixels, output_features)
#         self.beta = beta
#         self.isolate_superpixels = isolate_superpixels

#     def forward(self, x, segment):
#         segment = consecutive_reindex_batch_of_integers_tensor(segment)

#         fcolor, segment = extract_patch_around_superpixels(
#             x, segment, beta=self.beta, isolate_superpixel=self.isolate_superpixels
#         )
#         B, N, C, H, W = fcolor.shape
#         fcolor = fcolor.view(B * N, C, H, W)
#         features = self.model(fcolor)[-1]
#         features = self.last_conv(features)
#         F = features.shape[1]
#         f = features.view(B, N, F, -1).mean(dim=-1)

#         return f, segment


class SegmentScatteringCNN(BaseCNN):
    def forward(self, x, segment, mode="mean"):
        """
        x: (B, C, H, W)
        segment: (B, H, W)
        return features: (B, F, N)
               segment: (B, H, W)
        """
        b, h, w = segment.shape

        # with torch.amp.autocast("cuda"):
        features = self.model.forward(x)[self.f_index]
        # x_img = F.interpolate(x, size=features.shape[-2:], mode="bilinear", align_corners=False)
        # features = torch.cat([features, x_img], dim=1)

        segment = reindex_batch_segments(segment)
        if self.optimized:
            features = scatter_segments(features.float(), segment, mode, "bilinear")
        else:
            features = F.interpolate(features, size=(h, w), mode="bilinear", align_corners=False)
            features = einops.rearrange(features, "b c h w -> b (h w) c")
            segment = einops.rearrange(segment, "b h w -> b (h w)")
            features = scatter(features, segment, dim=1, reduce=mode)

            features = features.permute(0, 2, 1)
            print(features.shape)

        if self.cat_input:
            x = einops.rearrange(x, "b c h w -> b (c h w)")
            x = scatter(x, segment, dim=1, reduce=mode)
            segment = einops.rearrange(segment, "b h w -> b (h w)")
            f_img = scatter(x, segment, dim=1, reduce=mode)

            features = torch.cat([features.float(), f_img], dim=1)

        segment = segment.view(b, h, w)

        features = self.last_conv(features.float()).permute(0, 2, 1)

        # x_img = (x_img - x_img.min(dim=-2, keepdim=True).values) / (
        #     x_img.max(dim=-2, keepdim=True).values - x_img.min(dim=-2, keepdim=True).values + 1e-8
        # )
        features = self.act_fn(features)
        # features = F.normalize(features, p=2, dim=-1)

        # Features has shape (B, N, F)
        return features, segment


def test():
    from vitRet.data.data_factory import get_test_sample

    sample = get_test_sample(batch_size=16)

    image = sample["image"].cuda()
    segment = sample["segments"].cuda()
    model = SegmentScatteringCNN("hf_hub:ClementP/FundusDRGrading-seresnet50", feature_index=3).cuda()

    out = model(image, segment)
    print(out[0].shape)
    l = out[0].sum()
    l.backward()
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)

    # torch.cuda.synchronize()
    print(out[0].shape)


if __name__ == "__main__":
    test()
