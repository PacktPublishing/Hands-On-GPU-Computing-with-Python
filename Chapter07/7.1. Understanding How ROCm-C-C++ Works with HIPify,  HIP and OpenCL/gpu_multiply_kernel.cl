// GPU kernel function to multiply two array elements and also update the results on a third array
__kernel void multiply(__global double *p, __global double *q, __global double *r) {
 
    // Indexing the current element to process - equivalent to int index = threadIdx.x + blockIdx.x * blockDim.x in CUDA
    int index = get_global_id(0);
 
    // Simultaneous multiplication within this OpenCL kernel
    r[index] = p[index] * q[index];
}
