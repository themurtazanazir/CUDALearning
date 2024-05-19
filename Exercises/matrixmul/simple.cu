#include "utils.h"
using namespace std;

__global__
void simpleSquareMatrixMulKernel(float* out, float* m1, float* m2,  int size){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < size && col < size){
        float output = 0;
        for (int i=0; i<size; ++i){
            float e1 = m1[row*size+i]; //get (row, i) from m1
            float e2 = m2[i*size+col]; //get (i, col) from m2
            output += (e1*e2);
        }
        out[row*size+col] = output;
    }
}

__global__
void simpleMatrixMulKernel(float* out, float* m1, float* m2, int l, int m, int r){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < l && col < r){
        float output = 0;
        for (int i=0; i<m; ++i){
            float e1 = m1[row*m+i]; //get (row, i) from m1
            float e2 = m2[i*r+col]; //get (i, col) from m2
            // float e2 = m2[i * r + col]; // This line is incorrect //get (i, col) from m2
            output += (e1*e2);
        }
        out[row*r+col] = output;
    }
}

torch::Tensor simpleSquareMatrixMul(torch::Tensor m1, torch::Tensor m2) {
    CHECK_INPUT(m1);
    CHECK_INPUT(m2);
    int size = m1.size(0);
    auto output = torch::empty_like(m1);
    dim3 tpb(16, 16);
    dim3 blocks(cdiv(size, tpb.x), cdiv(size, tpb.y));
    // cout<<"size: "<<size<<endl;
    // cout<<"tbp: "<<tpb.x<<","<<tpb.y<<endl;
    // cout<<"blocks: "<<blocks.x<<","<<blocks.y<<endl;

    simpleSquareMatrixMulKernel<<<blocks, tpb>>>(
        output.data_ptr<float>(),
        m1.data_ptr<float>(),
        m2.data_ptr<float>(),
        size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor simpleMatrixMul(torch::Tensor m1, torch::Tensor m2) {
    CHECK_INPUT(m1);
    CHECK_INPUT(m2);
    int l = m1.size(0);
    int m = m1.size(1);
    int r = m2.size(1);
    auto output = torch::empty({l, r}, m1.options());
    dim3 tpb(16, 16);
    dim3 blocks(cdiv(r, tpb.x), cdiv(l, tpb.y));
    // cout<<"size: "<<size<<endl;
    cout<<"tbp: "<<tpb.x<<","<<tpb.y<<endl;
    cout<<"blocks: "<<blocks.x<<","<<blocks.y<<endl;
    cout<<"size: "<<output.size(0)<<","<<output.size(1)<<","<<endl;
    cout<<"l: "<<l<<", m: "<<m<<", r: "<<r<<endl;

    simpleMatrixMulKernel<<<blocks, tpb>>>(
        output.data_ptr<float>(),
        m1.data_ptr<float>(),
        m2.data_ptr<float>(),
        l, m, r);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
