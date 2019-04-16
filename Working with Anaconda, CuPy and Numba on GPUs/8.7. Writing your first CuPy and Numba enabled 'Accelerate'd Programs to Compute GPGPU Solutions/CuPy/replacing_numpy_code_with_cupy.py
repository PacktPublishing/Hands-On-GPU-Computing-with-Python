import cupy as cp #Importing CuPy
from timeit import default_timer as timer #To record computation time

N = 500000000 #500 million elements

#Starting timer to record GPU computation time
start = timer()

#Setting two arrays all with zero values
a_cp = cp.zeros(N).astype(cp.double)
b_cp = cp.zeros(N).astype(cp.double)

#Initializing two values for each array
a_cp.fill(23.0)
b_cp.fill(12.0)

#Updating second array as a product of both arrays
b_cp  = a_cp * b_cp

#Choosing a random index
random = cp.random.randint(0,N)

#Displaying random element
print("Choosing array element %d at random:" % random)
print("Random array element value is %lf" % b_cp[random])

#Displaying current GPU device
print(b_cp.device)

#GPU computation time
cupy_gpu_time = timer() - start

#Displaying GPU computation time
print("CuPy on GPU took %f seconds." % cupy_gpu_time)
