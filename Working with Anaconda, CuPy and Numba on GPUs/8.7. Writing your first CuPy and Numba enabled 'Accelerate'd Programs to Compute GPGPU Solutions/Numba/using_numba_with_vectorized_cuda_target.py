import numpy as np #Importing NumPy
from timeit import default_timer as timer #To record computation time
from numba import vectorize #Importing vectorize module from Numba

N = 500000000 #500 million elements

#The @vectorize decorator turns the function into a GPU-based vectorized function.
@vectorize(["double(double, double)"], target='cuda')
def vector_multiply_gpu(a, b):
    return a * b

def main():
    #Initialize the two arrays of double data type all with 0.0 values upto N
    p = np.zeros(N, dtype=np.double)
    q = np.zeros(N, dtype=np.double)

    #Update all the elements in the two arrays with 23.0 and 12.0 respectively
    p.fill(23.0)
    q.fill(12.0)

    # Time the GPU function
    start = timer()
    #Display computed array
    print(vector_multiply_gpu(p, q))
    vector_multiply_gpu_time = timer() - start

    #Report Computation Time
    print("GPU function took %f seconds." % vector_multiply_gpu_time)

    return 0


if __name__ == "__main__":
    main()
