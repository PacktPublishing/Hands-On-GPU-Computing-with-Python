import pyopencl as cl  # Importing the OpenCL API
import numpy  # Importing tools to work with numbers
from time import time  # Enabling access to the current time

N = 500000000 # 500 Million Elements

a = numpy.zeros(N).astype(numpy.double)  # Create a numpy array with all zeroes
b = numpy.zeros(N).astype(numpy.double)  # Create a second numpy array with all zeroes
c_gpu = numpy.empty_like(a)  # Creating an empty array the same size as array a to receive computed results from the GPU device

a.fill(23.0)  # set all values as 23
b.fill(12.0)  # set all values as 12

opencl_context = cl.create_some_context()  # Initialize the Context

command_queue = cl.CommandQueue(opencl_context, properties=cl.command_queue_properties.PROFILING_ENABLE)  # Instantiate a Queue and enable profiling to report computation time

a_buffer = cl.Buffer(opencl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)

b_buffer = cl.Buffer(opencl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)

c_buffer = cl.Buffer(opencl_context, cl.mem_flags.WRITE_ONLY, b.nbytes)  # Creating three memory buffers on the device (GPU)

opencl_kernel = """
// GPU kernel function to multiply two array elements and also update the results on a third array

__kernel void multiply(__global double *p, __global double *q, __global double *r) 
{
    // Indexing the current element to process - equivalent to int index = threadIdx.x + blockIdx.x * blockDim.x in CUDA

    int index = get_global_id(0);

    // Simultaneous multiplication within this OpenCL kernel

    r[index] = p[index] * q[index];

}"""  # OpenCL Kernel Creation: Note these are the exact same contents as we saw on our .cl file

opencl_program = cl.Program(opencl_context, opencl_kernel).build()

gpu_start_time = time()  # Get the GPU start time

event = opencl_program.multiply(command_queue, a.shape, None, a_buffer, b_buffer, c_buffer)  # Enqueue the multiply program on the GPU device

event.wait()  # Waiting until event completion

elapsed = 1e-6 * (event.profile.end - event.profile.start)  # Calculating execution time (Multiplying by 10^6 to get value in milliseconds from nanoseconds)

print("GPU Kernel Function took: {0} milliseconds".format(elapsed))  # Reporting kernel execution time

cl.enqueue_copy(command_queue, c_gpu, c_buffer).wait()  # Get back the data from GPU device memory into array c_gpu

gpu_end_time = time()  # Stores time point at the end of GPU computation

print("GPU Time(Inclusive of memory transfer between host and device (GPU)): {0} seconds".format(gpu_end_time - gpu_start_time))  # Reporting GPU execution time, including memory transfers between host and device

random = numpy.random.randint(0, N)

# Randomly choose index from second array to confirm changes to second array
print("New value of third array element with random index", random, "is", c_gpu[random])


