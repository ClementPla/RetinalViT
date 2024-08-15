#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS 512

template <typename scalar_t>
__global__ void superpixels_scatter_cuda_kernel(
    const __restrict__ torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> src, // [batch, channels, height, width]
    const __restrict__ torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> index,    // [batch, height, width]
    torch::PackedTensorAccessor32<scalar_t, 3> output,                                           // [batch, superpixel_id, channels, beta, beta]
    const scalar_t rescale_x, const scalar_t rescale_y)
{
    int batch_id = blockIdx.x;
    int channel_id = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_id >= src.size(0))
        return; // Check for invalid batch id

    if (channel_id >= src.size(1))
        return;
    int h = src.size(2);
    int w = src.size(3);
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            int index_x = static_cast<int>(x * rescale_x);
            int index_y = static_cast<int>(y * rescale_y);
            int superpixel_id = index[batch_id][index_y][index_x];
            if (superpixel_id < output.size(1))
            {
                atomicAdd(&output[batch_id][superpixel_id][channel_id], src[batch_id][channel_id][y][x]);
            }
        }
    }
}

torch::Tensor superpixels_scatter_sum_cuda(const torch::Tensor src, const torch::Tensor superpixels)
{

    auto options = torch::TensorOptions().dtype(src.dtype()).device(src.device());
    int superpixelsPerImage = superpixels.max().item<int>() + 1;

    torch::Tensor output = torch::zeros({src.size(0), src.size(1), superpixelsPerImage}, options);

    auto rescale_x = static_cast<float>(superpixels.size(2)) / static_cast<float>(src.size(3));
    auto rescale_y = static_cast<float>(superpixels.size(1)) / static_cast<float>(src.size(2));
    dim3 blocks(src.size(0), (src.size(1) + THREADS - 1) / THREADS);

    AT_DISPATCH_FLOATING_TYPES(src.type(), "superpixels_scatter_cuda_kernel", ([&]
                                                                               { superpixels_scatter_cuda_kernel<scalar_t><<<blocks, THREADS>>>(
                                                                                     src.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                                                                                     superpixels.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
                                                                                     output.packed_accessor32<scalar_t, 3>(),
                                                                                     rescale_x, rescale_y); }));

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::string errorMessage = "CUDA kernel failed with error: " + std::string(cudaGetErrorString(error));
        throw std::runtime_error(errorMessage);
    }

    return output;
}

template <typename scalar_t>
__global__ void superpixels_scatter_mean_cuda_kernel(
    const __restrict__ torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> src,
    const __restrict__ torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> index,
    torch::PackedTensorAccessor32<scalar_t, 3> output,
    torch::PackedTensorAccessor32<int, 2> counts,
    const scalar_t rescale_x, const scalar_t rescale_y)
{
    int batch_id = blockIdx.x;
    int channel_id = threadIdx.x + blockIdx.y * blockDim.x;
    if (batch_id >= src.size(0) || channel_id >= src.size(1))
        return;

    int h = src.size(2);
    int w = src.size(3);

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            int index_x = static_cast<int>(x * rescale_x);
            int index_y = static_cast<int>(y * rescale_y);
            int superpixel_id = index[batch_id][index_y][index_x];
            if (superpixel_id >= output.size(1))
                continue;

            atomicAdd(&output[batch_id][superpixel_id][channel_id], src[batch_id][channel_id][y][x]);
            if (channel_id == 0)
            {
                atomicAdd(&counts[batch_id][superpixel_id], 1);
            }
        }
    }
}

torch::Tensor superpixels_scatter_mean_cuda(const torch::Tensor src, const torch::Tensor superpixels)
{
    auto options = torch::TensorOptions().dtype(src.dtype()).device(src.device());
    int superpixelsPerImage = superpixels.max().item<int>() + 1;
    torch::Tensor output = torch::zeros({src.size(0), superpixelsPerImage, src.size(1)}, options);
    torch::Tensor counts = torch::zeros({src.size(0), superpixelsPerImage}, torch::TensorOptions().dtype(torch::kInt32).device(src.device()));

    auto rescale_x = static_cast<float>(superpixels.size(2)) / static_cast<float>(src.size(3));
    auto rescale_y = static_cast<float>(superpixels.size(1)) / static_cast<float>(src.size(2));

    dim3 blocks(src.size(0), (src.size(1) + THREADS - 1) / THREADS);

    AT_DISPATCH_FLOATING_TYPES(src.type(), "superpixels_scatter_mean_cuda_kernel", ([&]
                                                                                    { superpixels_scatter_mean_cuda_kernel<scalar_t><<<blocks, THREADS>>>(
                                                                                          src.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                                                                                          superpixels.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
                                                                                          output.packed_accessor32<scalar_t, 3>(),
                                                                                          counts.packed_accessor32<int, 2>(),
                                                                                          rescale_x, rescale_y); }));
    counts = counts.unsqueeze(2).to(output.dtype());
    counts = counts + (counts == 0).to(output.dtype());

    // Divide the sum by the count to get the mean
    output.div_(counts);

    return output;
}