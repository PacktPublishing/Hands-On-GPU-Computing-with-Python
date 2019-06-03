#Auto initialization for CUDA
import pycuda.autoinit
#Importing SourceModule from the PyCUDA Compiler module
from pycuda.compiler import SourceModule

#For a single thread on a single block
N=1
#Setting threads per block. Here, it is set to 1
Threads_per_block = (int(1))
#Setting blocks per grid, also calculated as 1.
Blocks_per_grid = (int((N + Threads_per_block - 1) / Threads_per_block))

#Printing from the GPU device itself!
mod = SourceModule("""
__global__ void hello_from_nvidia_gpu()
{
  printf("Hello World from NVIDIA GPU!");
}
""")
#Return the Function name in the get_function module
hello_from_nvidia_gpu = mod.get_function("hello_from_nvidia_gpu")

#Invoking the NVIDIA GPU Kernel
hello_from_nvidia_gpu(block=(Threads_per_block, 1, 1), grid=(Blocks_per_grid, 1))