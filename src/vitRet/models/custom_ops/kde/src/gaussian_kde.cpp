#include <torch/extension.h>
#include <iostream>
#include <cmath> // For pow and M_PI

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void gaussian_kde_cuda(const torch::Tensor points, const torch::Tensor sampling_points, 
                       const torch::Tensor mask, torch::Tensor out, float h, float Z, const bool is_2d);


torch::Tensor gaussian_kde_2d(const torch::Tensor points, const torch::Tensor sampling_points, 
                  const torch::Tensor mask, float h_coeff) {
    CHECK_INPUT(points);
    CHECK_INPUT(sampling_points);
    CHECK_INPUT(mask);
    auto mask_sum = mask.sum({1}); // Sum over the support dimensions
    float n_mean = mask_sum.to(torch::kFloat).mean().item<float>();

    float d = static_cast<float>(points.size(2));
    // Calculate h using the given formula
    float h = h_coeff * std::pow(n_mean * (d + 2) / 4.0, -1.0 / (d + 4));

    // Calculate Z using the given formula
    float Z = 2 * M_PI * h * h;
    auto output_tensor = torch::zeros({points.size(0), sampling_points.size(0)}, torch::kFloat32).to(points.device()).contiguous();
    gaussian_kde_cuda(points, sampling_points, mask.to(torch::kBool), output_tensor, h, Z, true);
    return output_tensor;
}

torch::Tensor gaussian_kde_1d(const torch::Tensor points, const torch::Tensor sampling_points, 
                  const torch::Tensor mask, float h_coeff) {
    CHECK_INPUT(points);
    CHECK_INPUT(sampling_points);
    CHECK_INPUT(mask);
    auto mask_sum = mask.sum({1}); // Sum over the support dimensions
    float n_mean = mask_sum.to(torch::kFloat).mean().item<float>();

    float d = static_cast<float>(points.size(2));
    // Calculate h using the given formula
    float h = h_coeff * std::pow(n_mean * (d + 2) / 4.0, -1.0 / (d + 4));

    float Z = std::pow(2 * M_PI, 0.5) * h;
    auto output_tensor = torch::zeros({points.size(0), sampling_points.size(0)}, torch::kFloat32).to(points.device()).contiguous();
    gaussian_kde_cuda(points, sampling_points, mask.to(torch::kBool), output_tensor, h, Z, false);
    return output_tensor;
}

torch::Tensor gaussian_kde(const torch::Tensor points, const torch::Tensor sampling_points, 
                  const torch::Tensor mask, float h_coeff) {
    if (points.size(2) == 1) {
        return gaussian_kde_1d(points, sampling_points, mask, h_coeff);
    } else {
        return gaussian_kde_2d(points, sampling_points, mask, h_coeff);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gaussian_kde", &gaussian_kde, "Gaussian KDE CUDA implementation");
}