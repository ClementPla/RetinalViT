from typing import Any

import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load
from vitRet.models.superpixels.indexing import reconstruct_spatial_map_from_segment
from vitRet.models.superpixels.topo import count_segment_occurence

superpixels_scatter = load(
    name="superpixels_scatter",
    sources=[
        "src/vitRet/models/custom_ops/superpixel_features_scatter/src/superpixels_scatter_cuda.cu",
        "src/vitRet/models/custom_ops/superpixel_features_scatter/src/superpixel_scatter.cpp",
    ],
    verbose=True,
    with_cuda=True,
    is_standalone=False,
    is_python_module=True,
)


# superpixels_scatter_backward = load(
#     name="superpixels_scatter_backward",
#     sources=[
#         "src/vitRet/models/custom_ops/superpixel_features_scatter/src/superpixel_scatter_backward_cuda.cu",
#         "src/vitRet/models/custom_ops/superpixel_features_scatter/src/superpixel_scatter_backward.cpp",
#     ],
#     verbose=True,
#     with_cuda=True,
#     is_standalone=False,
#     is_python_module=True,
# )


def extract_feature_in_superpixels(features, superpixels, reduce="mean"):
    result = superpixels_scatter.superpixels_scatter(features, superpixels, reduce)
    return result


class SuperpixelsScatter(Function):
    @staticmethod
    def forward(ctx, features, superpixels, reduce="mean"):
        ctx.reduce = reduce
        ctx.f_shape = features.shape[-2:]
        ctx.save_for_backward(superpixels)
        output = extract_feature_in_superpixels(features.contiguous(), superpixels, reduce)
        output = output.permute(0, 2, 1)
        output.requires_grad = features.requires_grad
        return output

    @staticmethod
    def backward(ctx: Any, grad_outputs) -> Any:
        assert ctx.reduce == "mean", 'Only "mean" reduce is supported for backward pass for now.'
        (superpixels,) = ctx.saved_tensors
        segment_counts = count_segment_occurence(superpixels)

        grad_outputs = grad_outputs / (segment_counts.unsqueeze(1) + 1e-7)
        if superpixels.ndim == 3:
            superpixels = superpixels.unsqueeze(1)
        grad_outputs
        small_superpixels = F.interpolate(superpixels.float(), size=ctx.f_shape, mode="nearest").squeeze(1).long()
        grad_outputs = reconstruct_spatial_map_from_segment(grad_outputs, small_superpixels, reindex=False)

        return grad_outputs, None, None


scatter_segments = SuperpixelsScatter.apply

if __name__ == "__main__":
    import torch

    img = torch.rand(1, 3, 64, 64).cuda()
    superpixels = torch.randint(0, 4, (1, 4, 4)).cuda()

    superpixels = (
        torch.nn.functional.interpolate(superpixels.float().unsqueeze(1), size=(256, 256), mode="nearest")
        .squeeze(1)
        .long()
    )

    result = extract_feature_in_superpixels(img.cuda(), superpixels.cuda())

    print(result.shape)
