import pyopencl as cl
import numpy as np
import pyopencl.array as clarray
from time import time
from pyopencl.elementwise import ElementwiseKernel

N = 500000000 #500 Million Elements

gpu_start_time = time()  # Noting GPU start time

#In contrast to PyCUDA, note that in PyOpenCL, we have to initialize a Context first.
pyopencl_context = cl.create_some_context()
#Now we instantiate a Command Queue with the PyOpenCL Context and also enable profiling to report computation time.
command_queue = cl.CommandQueue(pyopencl_context, properties=cl.command_queue_properties.PROFILING_ENABLE)

#Here, we multiply the two values and store the product on all the elements of a third array
#Note that we pass C syntax into the ElementwiseKernel. The difference here with PyCUDA is that we have
#to specify the Context first as an argument to the ElementWiseKernel
multiply = ElementwiseKernel(
        pyopencl_context,
        "double *a_gpu, double *b_gpu, double *c_gpu",
        "c_gpu[i] = a_gpu[i] / b_gpu[i]",
        "multiply")

#Initialize the first two arrays with random values and the third as zeroes
a_gpu = clarray.to_device(command_queue, np.random.rand(N).astype(np.double))
b_gpu = clarray.to_device(command_queue, np.random.rand(N).astype(np.double))
c_gpu = clarray.to_device(command_queue, np.zeros(N).astype(np.double))

event = multiply(a_gpu, b_gpu, c_gpu)

event.wait()  # Waiting until the event completes

elapsed = 1e-6 * (event.profile.end - event.profile.start)  # Calculating execution time (Multiplying by 10^6 to get value in milliseconds from nanoseconds)

print("GPU Kernel Function took {0} milliseconds".format(elapsed))  # Reporting kernel execution time

gpu_end_time = time()  # Get the GPU end time

print("GPU Time(Inclusive of memory transfer between host and device (GPU)): {0} seconds".format(gpu_end_time - gpu_start_time))  # Reporting GPU execution time, including memory transfers between host and device

random = np.random.randint(0,N)

#Randomly choose index from second array to confirm changes to the third array
print("New value of third array element with random index", random, "is", c_gpu[random])