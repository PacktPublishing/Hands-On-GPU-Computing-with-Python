import pyopencl as cl
import pyopencl.array as clarray
from time import time
import numpy as np

N = 500000000 #500 Million Elements

begin1 = time()

#In contrast to PyCUDA, note that in PyOpenCL, we have to initialize a Context first.
pyopencl_context = cl.create_some_context()
#Now we instantiate a Command Queue with the PyOpenCL Context
command_queue = cl.CommandQueue(pyopencl_context)

a_gpu = clarray.to_device(command_queue, np.zeros(N).astype(np.double))
a_gpu.fill(23.0)
begin2 = time()
b_gpu=a_gpu*12.0
end = time()

pyopencl_gpu_time = (end-begin2)/1e-3

total_gpu_time = end-begin1

random = np.random.randint(0,N)

print("\nGPU multiplication of array took %f milliseconds." % pyopencl_gpu_time)

print("\nGPU Time including dependent code %f seconds." % total_gpu_time)

print("\nChoosing second array element with index", random, "at random:", b_gpu[random])






