#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


template <typename scalar_t>
__global__ void gaussian_kde_kernel_2d(
    const torch::PackedTensorAccessor32<scalar_t, 3,torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t, 2,torch::RestrictPtrTraits> sampling_points,
    const torch::PackedTensorAccessor32<bool, 2,torch::RestrictPtrTraits> mask,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> out,
    const int support_dims, const int n_features, 
    const float h, const float Z) {
    
    // Batch index
    const int n = blockIdx.x;
    // column index
    const int c = blockIdx.y * blockDim.x + threadIdx.x;

    if (c >= out.size(1) || n >= out.size(0)) return; // Guard against out-of-bounds access

    scalar_t sum = 0.0;
    int n_element = 0;
    for (int i = 0; i < support_dims; i++) {
        if (!mask[n][i]) {
            continue;
        }
        n_element += 1;
        scalar_t diff = (points[n][i][0] - sampling_points[c][0]) * (points[n][i][0] - sampling_points[c][0]);
        diff += (points[n][i][1] - sampling_points[c][1]) * (points[n][i][1] - sampling_points[c][1]);
        sum += exp(-(diff) / (2.0 * h * h));
    }
    if (n_element == 0) {
        out[n][c] = sum;
        return;
    }
    out[n][c] = sum / (Z * static_cast<scalar_t>(n_element));
}

template <typename scalar_t>
__global__ void gaussian_kde_kernel_1d(
    const torch::PackedTensorAccessor32<scalar_t, 3,torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t, 2,torch::RestrictPtrTraits> sampling_points,
    const torch::PackedTensorAccessor32<bool, 2,torch::RestrictPtrTraits> mask,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> out,
    const int support_dims, const int n_features, 
    const float h, const float Z) {
    
    // Batch index
    const int n = blockIdx.x;
    // column index
    const int c = blockIdx.y * blockDim.x + threadIdx.x;

    if (c >= out.size(1) || n >= out.size(0)) return; // Guard against out-of-bounds access

    scalar_t sum = 0.0;
    int n_element = 0;
    for (int i = 0; i < support_dims; i++) {
        if (!mask[n][i]) {
            continue;
        }
        n_element += 1;
        scalar_t diff = points[n][i][0] - sampling_points[c][0];
        sum += exp(-(diff*diff) / (2.0 * h * h));
    }
    if (n_element == 0) {
        out[n][c] = sum;
        return;
    }
    out[n][c] = sum / (Z * static_cast<scalar_t>(n_element));
}

void gaussian_kde_cuda(const torch::Tensor points, const torch::Tensor sampling_points, 
                       const torch::Tensor mask, torch::Tensor out, float h, float Z, const bool is_2d) {
    

    const auto batch_size = points.size(0);
    const auto support_dims = points.size(1);
    const auto n_features = points.size(2);
    const auto n_sampling_points = sampling_points.size(0);
    const int threads = 1024;
    const dim3 blocks(batch_size, (n_sampling_points + threads - 1) / threads);
    if (is_2d) {
    AT_DISPATCH_FLOATING_TYPES(points.type(), "gaussian_kde_cuda", ([&] {
        gaussian_kde_kernel_2d<<<blocks, threads>>>(
            points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            sampling_points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            mask.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
            out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            support_dims, n_features, h, Z);
    }));
    }
    else {
    AT_DISPATCH_FLOATING_TYPES(points.type(), "gaussian_kde_cuda", ([&] {
        gaussian_kde_kernel_1d<<<blocks, threads>>>(
            points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            sampling_points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            mask.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
            out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            support_dims, n_features, h, Z);
    }));
    };
    

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::string errorMessage = "CUDA kernel failed with error: " + std::string(cudaGetErrorString(error));
        throw std::runtime_error(errorMessage);
    }
};
