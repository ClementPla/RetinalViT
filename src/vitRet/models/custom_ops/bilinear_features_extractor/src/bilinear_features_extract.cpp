#include <torch/extension.h>
#include <iostream>
#include <cmath> // For pow and M_PI

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor bilinear_features_extract_cuda(const torch::Tensor features, const torch::Tensor superpixels,
                                             const torch::Tensor boundingBoxes, const int beta, const bool isolateSuperpixels);

torch::Tensor bilinear_features_extract(const torch::Tensor features, const torch::Tensor superpixels,
                                        const torch::Tensor boundingBoxes, const int beta, const bool isolateSuperpixels)
{
    CHECK_INPUT(features);
    CHECK_INPUT(superpixels);
    CHECK_INPUT(boundingBoxes);
    auto superpixelsInt = superpixels.to(torch::kInt);
    auto boundingBoxesInt = boundingBoxes.to(torch::kInt);

    return bilinear_features_extract_cuda(features, superpixelsInt, boundingBoxesInt, beta, isolateSuperpixels);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bilinear_features_extract", &bilinear_features_extract, "Bilinear sampling of features in a bounding box defined by a segment");
}