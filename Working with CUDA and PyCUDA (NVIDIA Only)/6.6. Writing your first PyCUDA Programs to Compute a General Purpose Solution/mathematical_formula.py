import pycuda.autoinit
import numpy
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
from pycuda.elementwise import ElementwiseKernel

#Here, we compute a mathematical formula
#Note that we pass C/C++ syntax into the ElementwiseKernel
compute = ElementwiseKernel(
        "double *a_gpu, double *b_gpu, double *c_gpu",
        "c_gpu[i] = (cos(a_gpu[i])*cos(a_gpu[i])) * (b_gpu[i]* b_gpu[i])+sin(a_gpu[i]*b_gpu[i])",
        "compute")

N = 500000000
#Threads_per_block = (int(1024))
#Blocks_per_grid = (int((N + Threads_per_block - 1) / Threads_per_block))

a_gpu = gpuarray.to_gpu(numpy.zeros(N).astype(numpy.double))
b_gpu = gpuarray.to_gpu(numpy.zeros(N).astype(numpy.double))
c_gpu = gpuarray.to_gpu(numpy.zeros(N).astype(numpy.double))

a_gpu.fill(24.0)
b_gpu.fill(12.0)

start = drv.Event()
end = drv.Event()

# Time the GPU function
start.record()
compute(a_gpu, b_gpu, c_gpu)
end.record()
end.synchronize()
gpu_compute_time = start.time_till(end)

random = numpy.random.randint(0,N)

#Randomly choose index from second array to confirm changes to second array
print("New value of second array element with random index ", random, "is ", c_gpu[random])

# Report times
print("GPU function took %f milliseconds." % gpu_compute_time)