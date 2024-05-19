#include "utils.h"

#define TILE_SIZE 16


using namespace std;

__global__
void tiledSquareMatrixMulKernel(float *out, float *m1, float *m2, int width){
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    __shared__ float sharedM1[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedM2[TILE_SIZE][TILE_SIZE];

    float output = 0;
    for (int tile_offset=0; tile_offset<width; tile_offset+=TILE_SIZE){
        // each thread will grab (row, threadIdx.x+tile_offset) from m1
        sharedM1[threadIdx.y][threadIdx.x] = m1[row*width+threadIdx.x+tile_offset];
        // each thread will grab (threadIdx.y+tile_offset, col) from m2
        sharedM2[threadIdx.y][threadIdx.x] = m2[(threadIdx.y+tile_offset)*width + col];
        __syncthreads();

        for (int i=0; i<TILE_SIZE; i++){
            output += (sharedM1[threadIdx.y][i]*sharedM2[i][threadIdx.x]);
        }
        __syncthreads();
    }
    out[row*width+col] = output;

}

__global__ 
void tiledMatrixMulKernel(float* out, float* m1, float* m2, int l, int m, int r){
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    __shared__ float sharedM1[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedM2[TILE_SIZE][TILE_SIZE];

    float output = 0;
    for (int tile_offset=0; tile_offset<m; tile_offset+=TILE_SIZE){
        // each thread will grab (row, threadIdx.x+tile_offset) from m1
        if (row < l && threadIdx.x+tile_offset < m)
            sharedM1[threadIdx.y][threadIdx.x] = m1[row*m+threadIdx.x+tile_offset];
        else
            sharedM1[threadIdx.y][threadIdx.x] = 0.0f;

        // each thread will grab (threadIdx.y+tile_offset, col) from m2
        if ((threadIdx.y+tile_offset) < m && col < r)
            sharedM2[threadIdx.y][threadIdx.x] = m2[(threadIdx.y+tile_offset)*r + col];
        else
            sharedM2[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i=0; i<TILE_SIZE; i++){
            output += (sharedM1[threadIdx.y][i]*sharedM2[i][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < m && col < r)
        out[row*r+col] = output;



}

torch::Tensor tiledSquareMatrixMul(torch::Tensor m1, torch::Tensor m2) {
    CHECK_INPUT(m1);
    CHECK_INPUT(m2);
    int size = m1.size(0);
    auto output = torch::empty_like(m1);
    dim3 tpb(TILE_SIZE, TILE_SIZE);
    dim3 blocks(cdiv(size, tpb.x), cdiv(size, tpb.y));
    // cout<<"size: "<<size<<endl;
    // cout<<"tbp: "<<tpb.x<<","<<tpb.y<<endl;
    // cout<<"blocks: "<<blocks.x<<","<<blocks.y<<endl;

    tiledSquareMatrixMulKernel<<<blocks, tpb>>>(
        output.data_ptr<float>(),
        m1.data_ptr<float>(),
        m2.data_ptr<float>(),
        size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor tiledMatrixMul(torch::Tensor m1, torch::Tensor m2) {
    CHECK_INPUT(m1);
    CHECK_INPUT(m2);
    int l = m1.size(0);
    int m = m1.size(1);
    int r = m2.size(1);
    auto output = torch::empty({l, r}, m1.options());
    dim3 tpb(TILE_SIZE, TILE_SIZE);
    dim3 blocks(cdiv(r, tpb.x), cdiv(l, tpb.y));
    // cout<<"size: "<<size<<endl;
    // cout<<"tbp: "<<tpb.x<<","<<tpb.y<<endl;
    // cout<<"blocks: "<<blocks.x<<","<<blocks.y<<endl;

    tiledMatrixMulKernel<<<blocks, tpb>>>(
        output.data_ptr<float>(),
        m1.data_ptr<float>(),
        m2.data_ptr<float>(),
        l, m, r);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}