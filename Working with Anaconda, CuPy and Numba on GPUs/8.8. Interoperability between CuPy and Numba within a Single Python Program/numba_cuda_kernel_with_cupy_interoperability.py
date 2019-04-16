from numba import cuda #Using Numba
import cupy as cp #Using CuPy
from timeit import default_timer as timer
N = 500000000
@cuda.jit
def multiply(p, q):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    index = tx + ty * bw
    if index < N:  # Check array size limit
        q[index]=p[index]*q[index]

def main():
    a_source = cp.zeros(N, dtype=cp.double)
    b_source = cp.zeros(N, dtype=cp.double)

    a_source.fill(23)
    b_source.fill(12)

    # Time the GPU function
    threadsperblock = 1024
    blockspergrid = (N + (threadsperblock - 1)) // threadsperblock
    start = timer()
    multiply[blockspergrid, threadsperblock](a_source,b_source)
    vector_multiply_gpu_time = timer() - start

    # Report times
    print("GPU function took %f seconds." % vector_multiply_gpu_time)
    random = cp.random.randint(0,N)

    print("Choosing array element %d at random:" % random)
    print("Random array element value is %lf" % b_source[random])

if __name__ == "__main__":
    main()
