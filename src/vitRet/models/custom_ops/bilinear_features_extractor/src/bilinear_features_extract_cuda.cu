#include <torch/torch.h>
#include <iostream>

__device__ bool bound_check(int batch_id, int superpixel_id, int channel_id, int nb_superpixels, int nb_channels, int nb_batches)
{
    return batch_id >= 0 && superpixel_id >= 0 && superpixel_id < nb_superpixels && channel_id >= 0 && channel_id < nb_channels && batch_id < nb_batches;
}

__global__ void interpolate_superpixels_kernel_with_border(
    torch::PackedTensorAccessor32<float, 4> images, // [batch, channels, height, width]
    torch::PackedTensorAccessor32<int, 3> masks,    // [batch, height, width]
    torch::PackedTensorAccessor32<float, 5> output, // [batch, superpixel_id, channels, beta, beta]
    torch::PackedTensorAccessor32<int, 3> bboxes,   // [batch, nb_superpixels, 4]
    const int beta, const float rescale_x, const float rescale_y)
{
    int batch_id = blockIdx.x;
    int superpixel_id = threadIdx.x + blockIdx.y * blockDim.x;
    int channel_id = blockIdx.z;

    bool bounds = bound_check(batch_id, superpixel_id, channel_id, bboxes.size(1), images.size(1), images.size(0));
    if (!bounds)
        return;

    int xmin = bboxes[batch_id][superpixel_id][0] / rescale_x;
    int ymin = bboxes[batch_id][superpixel_id][1] / rescale_y;
    int xmax = bboxes[batch_id][superpixel_id][2] / rescale_x;
    int ymax = bboxes[batch_id][superpixel_id][3] / rescale_y;

    if (ymax <= 0 || xmax <= 0)
        return; // Check for invalid bounding boxes

    float scaleY = static_cast<float>(ymax - ymin + 1) / beta;
    float scaleX = static_cast<float>(xmax - xmin + 1) / beta;
    for (int by = 0; by < beta; ++by)
    {
        for (int bx = 0; bx < beta; ++bx)
        {
            float y = ymin + by * scaleY;
            float x = xmin + bx * scaleX;
            int y0_mask = max(0, min(static_cast<int>(y * rescale_x), masks.size(1) - 1));
            int x0_mask = max(0, min(static_cast<int>(x * rescale_y), masks.size(2) - 1));

            if (masks[batch_id][y0_mask][x0_mask] != superpixel_id)
            {
                continue;
            }
            int y0 = min(max(0, static_cast<int>(y)), images.size(2) - 1);
            int x0 = min(max(0, static_cast<int>(x)), images.size(3) - 1);
            float ya = y - y0, xa = x - x0;

            int y1 = min(y0 + 1, images.size(2) - 1);
            int x1 = min(x0 + 1, images.size(3) - 1);

            output[batch_id][superpixel_id][channel_id][by][bx] =
                images[batch_id][channel_id][y0][x0] * (1 - xa) * (1 - ya) +
                images[batch_id][channel_id][y1][x0] * (1 - xa) * ya +
                images[batch_id][channel_id][y0][x1] * xa * (1 - ya) +
                images[batch_id][channel_id][y1][x1] * xa * ya;
        }
    }
}

__global__ void interpolate_superpixels_kernel_without_border(
    torch::PackedTensorAccessor32<float, 4> images, // [batch, channels, height, width]
    torch::PackedTensorAccessor32<int, 3> masks,    // [batch, height, width]
    torch::PackedTensorAccessor32<float, 5> output, // [batch, superpixel_id, channels, beta, beta]
    torch::PackedTensorAccessor32<int, 3> bboxes,   // [batch, nb_superpixels, 4]
    const int beta, const float rescale_x, const float rescale_y)
{
    int batch_id = blockIdx.x;
    int superpixel_id = threadIdx.x + blockIdx.y * blockDim.x;
    int channel_id = blockIdx.z;

    bool bounds = bound_check(batch_id, superpixel_id, channel_id, bboxes.size(1), images.size(1), images.size(0));
    if (!bounds)
        return;

    int xmin = bboxes[batch_id][superpixel_id][0] / rescale_x;
    int ymin = bboxes[batch_id][superpixel_id][1] / rescale_y;
    int xmax = bboxes[batch_id][superpixel_id][2] / rescale_x;
    int ymax = bboxes[batch_id][superpixel_id][3] / rescale_y;

    if (ymax <= 0 || xmax <= 0)
        return; // Check for invalid bounding boxes

    float scaleY = static_cast<float>(ymax - ymin + 1) / beta;
    float scaleX = static_cast<float>(xmax - xmin + 1) / beta;

    for (int by = 0; by < beta; ++by)
    {
        for (int bx = 0; bx < beta; ++bx)
        {
            float y = ymin + by * scaleY;
            float x = xmin + bx * scaleX;
            int y0 = min(max(0, static_cast<int>(y)), images.size(2) - 1);
            int x0 = min(max(0, static_cast<int>(x)), images.size(3) - 1);
            int x0_mask = min(static_cast<int>(x * rescale_x), masks.size(2) - 1);
            int y0_mask = min(static_cast<int>(y * rescale_y), masks.size(1) - 1);

            int y1 = min(y0 + 1, images.size(2) - 1);
            int x1 = min(x0 + 1, images.size(3) - 1);

            float ya = y - y0, xa = x - x0;
            output[batch_id][superpixel_id][channel_id][by][bx] =
                images[batch_id][channel_id][y0][x0] * (1 - xa) * (1 - ya) +
                images[batch_id][channel_id][y1][x0] * (1 - xa) * ya +
                images[batch_id][channel_id][y0][x1] * xa * (1 - ya) +
                images[batch_id][channel_id][y1][x1] * xa * ya;
        }
    }
}

torch::Tensor bilinear_features_extract_cuda(const torch::Tensor images, const torch::Tensor superpixels,
                                             const torch::Tensor boundingBoxes, const int beta,
                                             const bool isolateSuperpixels)
{

    auto options = torch::TensorOptions().dtype(images.dtype()).device(images.device());
    float min_val = images.min().item<float>();
    torch::Tensor output = torch::full({images.size(0), boundingBoxes.size(1), images.size(1), beta, beta}, min_val, options);

    auto images_acc = images.packed_accessor32<float, 4>();
    auto superpixels_acc = superpixels.packed_accessor32<int, 3>();
    auto bboxes_acc = boundingBoxes.packed_accessor32<int, 3>();
    auto output_acc = output.packed_accessor32<float, 5>();
    int superpixelsPerImage = boundingBoxes.size(1);
    int threads = 1024;
    float rescale_x = static_cast<float>(superpixels.size(2)) / static_cast<float>(images.size(3));
    float rescale_y = static_cast<float>(superpixels.size(1)) / static_cast<float>(images.size(2));
    dim3 blocks(images.size(0), (superpixelsPerImage + threads - 1) / threads, images.size(1));
    if (isolateSuperpixels)
    {
        interpolate_superpixels_kernel_with_border<<<blocks, threads>>>(
            images_acc, superpixels_acc, output_acc, bboxes_acc, beta, rescale_x, rescale_y);
    }
    else
    {
        interpolate_superpixels_kernel_without_border<<<blocks, threads>>>(
            images_acc, superpixels_acc, output_acc, bboxes_acc, beta, rescale_x, rescale_y);
    }

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::string errorMessage = "CUDA kernel failed with error: " + std::string(cudaGetErrorString(error));
        throw std::runtime_error(errorMessage);
    }

    return output;
}