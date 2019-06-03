import cupy as cp #Importing CuPy

#Defining the CUDA kernel
multiply = cp.RawKernel(r'''
extern "C" __global__
void multiply(const int* p, const int* q, int* z) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    z[tid] = p[tid] * q[tid];
 }
''', 'multiply')

#First two arrays are set as 0,1,2,3....upto 300
p = cp.arange(300, dtype=cp.int)
q = cp.arange(300, dtype=cp.int)

#Setting a new array with zeros to pass to kernel for computation
z = cp.zeros(300, dtype=cp.int)

#Invoking the kernel with a grid of 250 blocks, each consisting of 1024 threads
multiply((250,1), (1024,1), (p, q, z))  # grid, block and arguments

#Displaying the output computed on the kernel
print(z)

