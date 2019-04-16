import numpy as np #Importing NumPy
from numba import njit #Importing njit module from Numba
from timeit import default_timer as timer #To record computation time

N = 500000000 #500 Million Elements

# Function to multiply two array elements and also update the results on the second array
@njit #Equivalent to @jit(nopython=True)
def multiply(p_cpu, q_cpu):
    for i in range(N):
        q_cpu[i] = p_cpu[i] * q_cpu[i]

def main():
    #Initialize the two arrays of double data type all with 0.0 values upto N
    p = np.zeros(N, dtype=np.double)
    q = np.zeros(N, dtype=np.double)
    #Update all the elements in the two arrays with 23.0 and 12.0 respectively
    p.fill(23.0)
    q.fill(12.0)

    #Time the Function
    begin = timer()
    multiply(p, q)
    numpy_cpu_time = timer() - begin

    #Report Computation Time
    print("Function took %f seconds." % numpy_cpu_time)

    #Choose a random integer index value between 0 to N
    random = np.random.randint(0, N)

    #Verify all values to be 276.0 for second array by random selection
    print("New value of second array element with random index", random, "is", q[random])

    return 0

if __name__ == "__main__":
    main()
