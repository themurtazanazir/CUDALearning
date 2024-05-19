#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <stdio.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

unsigned int cdiv(unsigned int a, unsigned int b);
