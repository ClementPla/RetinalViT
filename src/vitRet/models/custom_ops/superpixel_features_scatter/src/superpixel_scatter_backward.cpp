#include <torch/extension.h>
#include <iostream>
#include <cmath> // For pow and M_PI

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor superpixels_scatter_backward_cuda(const torch::Tensor &grad_output,
                                                const torch::Tensor &features,
                                                const torch::Tensor &superpixels,
                                                const std::string &reduce);

torch::Tensor superpixels_scatter_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &features,
    const torch::Tensor &superpixels,
    const std::string &reduce)
{
    TORCH_CHECK(features.dim() == 4, "features must be a 4D tensor");
    TORCH_CHECK(superpixels.dim() == 3, "superpixels must be a 3D tensor");
    TORCH_CHECK(features.size(0) == superpixels.size(0), "Batch sizes must match");
    CHECK_INPUT(features);
    CHECK_INPUT(superpixels);
    auto superpixelsInt = superpixels.to(torch::kInt);

    return superpixels_scatter_backward_cuda(grad_output, features, superpixelsInt, reduce);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("superpixels_scatter_backward", &superpixels_scatter_backward, "Backward pass of the extraction of features in a segment by upsampling+scattering");
}