from pathlib import Path
import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline
from time import time

def compile_extension_2d():
    cuda_source = Path("mean_filter_kernel_2d.cu").read_text()
    cpp_source = """torch::Tensor mean_filter_2d(torch::Tensor image, int radius);"""

    # Load the CUDA kernel as a PyTorch extension
    ext = load_inline(
        name="mean_filter_extension_2d",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["mean_filter_2d"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return ext

def compile_extension_1d():
    cuda_source = Path("mean_filter_kernel_1d.cu").read_text()
    cpp_source = """torch::Tensor mean_filter_1d(torch::Tensor image, int radius);"""

    # Load the CUDA kernel as a PyTorch extension
    ext = load_inline(
        name="mean_filter_extension_1d",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["mean_filter_1d"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return ext


def main():
    """
    Use torch cpp inline extension function to compile the kernel in mean_filter_kernel.cu.
    Read input image, convert apply mean filter custom cuda kernel and write result out into output.png.
    """
    ext_2d = compile_extension_2d()
    ext_1d = compile_extension_1d()

    x = read_image("grace_hq_gray.jpg").contiguous().squeeze().cuda()
    print(x.shape)
    assert x.dtype == torch.uint8
    print("Input image:", x.shape, x.dtype)

    
    t= time()
    for _ in range(200):
        y2 = ext_2d.mean_filter_2d(x, 8).cpu()
    t2 = time()
    for _ in range(200):
        y1 = ext_1d.mean_filter_1d(x, 8).cpu()
    t3 = time()
    # print("Output image:", y.shape, y.dtype)
    write_png(y2.unsqueeze(0), "output_2d.png")
    write_png(y1.unsqueeze(0), "output_1d.png")
    print("2D filter time:", t2-t)
    print("1D filter time:", t3-t2)


if __name__ == "__main__":
    main()