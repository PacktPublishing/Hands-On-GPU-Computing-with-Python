import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import pycuda.driver as cudrv

N = 500000000

begin = cudrv.Event()
end = cudrv.Event()

begin.record()
a_gpu = gpuarray.to_gpu(np.zeros(N).astype(np.double))
a_gpu.fill(23.0)
b_gpu=a_gpu*12.0
end.record()
end.synchronize()
pycuda_gpu_time = begin.time_till(end)

random = np.random.randint(0,N)

print("Choosing second array element with index", random, "at random:", b_gpu[random])

print("\nGPU took %f milliseconds." % pycuda_gpu_time)





