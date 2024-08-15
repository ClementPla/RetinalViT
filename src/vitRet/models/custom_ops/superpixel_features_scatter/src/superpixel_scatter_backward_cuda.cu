#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void superpixels_scatter_backward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> superpixels,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_input,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> counts,
    const scalar_t rescale_x, const scalar_t rescale_y,
    bool is_mean)
{
    int batch_id = blockIdx.x;
    int channel_id = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_id >= grad_input.size(0) || channel_id >= grad_input.size(1))
        return;

    int h = grad_input.size(2);
    int w = grad_input.size(3);

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            int superpixels_x = static_cast<int>(x * rescale_x);
            int superpixels_y = static_cast<int>(y * rescale_y);
            int superpixel_id = superpixels[batch_id][superpixels_y][superpixels_x];

            scalar_t grad_scale = 1.0;
            if (is_mean)
            {
                grad_scale = 1.0 / static_cast<scalar_t>(counts[batch_id][superpixel_id]);
            }

            grad_input[batch_id][channel_id][y][x] = grad_output[batch_id][superpixel_id][channel_id] * grad_scale;
        }
    }
}

torch::Tensor superpixels_scatter_backward_cuda(
    const torch::Tensor &grad_output,
    const torch::Tensor &features,
    const torch::Tensor &superpixels,
    const std::string &reduce)
{
    auto grad_input = torch::zeros_like(features);
    auto superpixelsInt = superpixels.to(torch::kInt);

    auto rescale_x = static_cast<float>(superpixels.size(2)) / static_cast<float>(features.size(3));
    auto rescale_y = static_cast<float>(superpixels.size(1)) / static_cast<float>(features.size(2));

    bool is_mean = (reduce == "mean");

    torch::Tensor counts;
    if (is_mean)
    {
        counts = torch::zeros({features.size(0), grad_output.size(1)},
                              torch::TensorOptions().dtype(torch::kInt32).device(features.device()));

        // Compute counts
        AT_DISPATCH_INTEGRAL_TYPES(superpixelsInt.scalar_type(), "compute_counts", ([&]
                                                                                    {
            auto superpixels_accessor = superpixelsInt.packed_accessor32<int, 3, torch::RestrictPtrTraits>();
            auto counts_accessor = counts.packed_accessor32<int, 2, torch::RestrictPtrTraits>();
            
            for (int b = 0; b < superpixels.size(0); ++b) {
                for (int h = 0; h < superpixels.size(1); ++h) {
                    for (int w = 0; w < superpixels.size(2); ++w) {
                        atomicAdd(&counts_accessor[b][superpixels_accessor[b][h][w]], 1);
                    }
                }
            } }));
    }

    dim3 blocks(features.size(0), (features.size(1) + THREADS - 1) / THREADS);
    dim3 threads(THREADS);

    AT_DISPATCH_FLOATING_TYPES(features.type(), "superpixels_scatter_backward_cuda", ([&]
                                                                                      { superpixels_scatter_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                                                                                            grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                                                                            superpixelsInt.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
                                                                                            grad_input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                                                                                            is_mean ? counts.packed_accessor32<int, 2, torch::RestrictPtrTraits>() : torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>(),
                                                                                            rescale_x, rescale_y,
                                                                                            is_mean); }));

    return grad_input;
}