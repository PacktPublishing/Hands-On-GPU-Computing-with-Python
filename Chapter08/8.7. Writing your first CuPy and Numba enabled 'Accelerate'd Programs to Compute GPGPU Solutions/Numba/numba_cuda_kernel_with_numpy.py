from numba import cuda #Importing cuda module from Numba
import numpy as np #Importing NumPy
from timeit import default_timer as timer #To record computation time
N = 500000000 #500 million elements

@cuda.jit
def multiply(p, q):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Number of threads per block
    bw = cuda.blockDim.x
    # Computing flattened index inside the array
    index = tx + ty * bw
    if index < N:  # Check array size limit
        q[index]=p[index]*q[index]

def main():
    #Initialize the two arrays of double data type all with 0.0 values upto N
    a_source = np.zeros(N, dtype=np.double)
    b_source = np.zeros(N, dtype=np.double)

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
    random = np.random.randint(0,N)

    print("Choosing array element %d at random:" % random)
    print("Random array element value is %lf" % b_source[random])

if __name__ == "__main__":
    main()
