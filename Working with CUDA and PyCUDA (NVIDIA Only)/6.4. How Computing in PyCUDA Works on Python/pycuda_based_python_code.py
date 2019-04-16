import pycuda.autoinit
import pycuda.driver as cudadrv
import numpy
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel

#Here, we multiply the two values and update all the elements of the second array with the new product
#Note that we pass C syntax into the ElementwiseKernel
multiply = ElementwiseKernel(
        "double *a_gpu, double *b_gpu",
        "b_gpu[i] = a_gpu[i] * b_gpu[i]",
        "multiply")

N = 500000000 #500 Million Elements

a_gpu = gpuarray.to_gpu(numpy.zeros(N).astype(numpy.double))
b_gpu = gpuarray.to_gpu(numpy.zeros(N).astype(numpy.double))

a_gpu.fill(23.0)
b_gpu.fill(12.0)

begin = cudadrv.Event()
end = cudadrv.Event()

# Time the GPU function
begin.record()
multiply(a_gpu, b_gpu)
end.record()
end.synchronize()
gpu_multiply_time = begin.time_till(end)

random = numpy.random.randint(0,N)

#Randomly choose index from second array to confirm changes to second array
print("New value of second array element with random index", random, "is", b_gpu[random])

# Report GPU Function time
print("GPU function took %f milliseconds." % gpu_multiply_time)