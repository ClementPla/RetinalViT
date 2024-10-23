from typing import Any

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

from vitRet.models.superpixels.indexing import reconstruct_spatial_map_from_segment
from vitRet.models.superpixels.topo import count_segment_occurence

superpixels_scatter = load(
    name="superpixels_scatter",
    sources=[
        "src/vitRet/models/custom_ops/superpixel_features_scatter/src/superpixels_scatter_nearest_cuda.cu",
        "src/vitRet/models/custom_ops/superpixel_features_scatter/src/superpixels_scatter_bilinear_cuda.cu",
        "src/vitRet/models/custom_ops/superpixel_features_scatter/src/superpixel_scatter.cpp",
    ],
    verbose=True,
    with_cuda=True,
    is_standalone=False,
    is_python_module=True,
)


class SuperpixelsScatter(Function):
    @staticmethod
    def forward(ctx, features, superpixels, reduce="mean", mode="bilinear", autobroadcast=True) -> torch.Tensor:
        features = features.contiguous()
        ctx.reduce = reduce
        ctx.f_shape = features.shape[-2:]
        ctx.save_for_backward(superpixels)
        output = superpixels_scatter.superpixels_scatter(features, superpixels, reduce, mode, autobroadcast)
        output = output.permute(0, 2, 1)
        return output

    @staticmethod
    def backward(ctx: Any, grad_outputs) -> Any:
        assert ctx.reduce in ["mean", "sum"], 'Only "mean" or "sum" reduce is supported for backward pass for now.'
        superpixels = ctx.saved_tensors[0].clone()

        if ctx.reduce == "mean":
            segment_counts = count_segment_occurence(superpixels)
            grad_outputs = grad_outputs / (segment_counts.unsqueeze(1) + 1e-10)
        if superpixels.ndim == 3:
            superpixels.unsqueeze_(1)
        small_superpixels = F.interpolate(superpixels.float(), size=ctx.f_shape, mode="nearest").squeeze(1).long()
        grad_outputs = reconstruct_spatial_map_from_segment(grad_outputs, small_superpixels, reindex=False)

        return grad_outputs, None, None, None


scatter_segments = SuperpixelsScatter.apply
