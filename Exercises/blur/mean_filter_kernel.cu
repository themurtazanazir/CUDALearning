#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <stdio.h>

using namespace std;

__global__ void mean_filter_kernel(unsigned char *output, unsigned char *input, int width, int height, int radius)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((row < height) && (col < width))
    {
        int val = 0;
        int count = 0;
        for (int blurRow = -radius; blurRow < radius; ++blurRow)
        {
            for (int blurCol = -radius; blurCol < radius; ++blurCol)
            {
                int i = row + blurRow;
                int j = col + blurCol;
                if (i < height && j < width && i >= 0 && j >= 0)
                {
                    val += input[width * i + j];
                    count++;
                }
            }
        }
        output[row * width + col] = (unsigned char)(val / count);
    }
}

torch::Tensor mean_filter(torch::Tensor image, int radius)
{
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);
    assert(radius > 0);

    const auto height = image.size(0);
    const auto width = image.size(1);

    auto result = torch::empty_like(image);

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        ceil((float)width / threads_per_block.x), ceil((float)height / threads_per_block.y));
    cout<<blocks_per_grid.x<<" "<<blocks_per_grid.y<<endl;
    cout<<threads_per_block.x<<" "<<threads_per_block.y<<endl;
    mean_filter_kernel<<<blocks_per_grid, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height,
        radius);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}