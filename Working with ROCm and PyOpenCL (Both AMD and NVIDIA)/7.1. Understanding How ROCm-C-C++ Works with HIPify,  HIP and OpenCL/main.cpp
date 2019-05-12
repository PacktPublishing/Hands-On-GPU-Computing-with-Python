#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>
#include <iostream>
using namespace std;

#define N (0x15000000) //15 Million Elements
 
int main(void) {
    // Creating two input arrays with two values on each
    double *p = (double*)malloc(sizeof(double)*N);
    double *q = (double*)malloc(sizeof(double)*N);
    for(int i = 0; i < N-1; ++i) {
        p[i] = 23;
        q[i] = 12;
    }
 
    // Loading source code from .cl file into the array cl_source
    FILE *opencl_file;
    char *cl_source;
    size_t source_size; 

    opencl_file = fopen("gpu_multiply_kernel.cl", "r");
    if (!opencl_file) {
        fprintf(stderr, "Failed to load opencl_kernel.\n");
        exit(1);
    }
    cl_source = (char*)malloc(N);
    source_size = fread( cl_source, 1, N, opencl_file);
    fclose( opencl_file );
 
    // Fetching platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint get_num_devices;
    cl_uint get_num_platforms;
    cl_int for_kernel = clGetPlatformIDs(1, &platform_id, &get_num_platforms);
    for_kernel = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &get_num_devices);
 
    // Creating an OpenCL context
    cl_context opencl_context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &for_kernel);
 
    // Creating a command queue and enabling profiling to find computation time

    cl_command_queue_properties profiling_on[] {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(opencl_context, device_id, profiling_on, &for_kernel);
    
    // Creating memory buffers on the device for each array - similar to cudaMalloc on CUDA 
    cl_mem p_gpu = clCreateBuffer(opencl_context, CL_MEM_READ_ONLY, N * sizeof(double), NULL, &for_kernel);
    cl_mem q_gpu = clCreateBuffer(opencl_context, CL_MEM_READ_ONLY, N * sizeof(double), NULL, &for_kernel);
    cl_mem r_gpu = clCreateBuffer(opencl_context, CL_MEM_WRITE_ONLY, N * sizeof(double), NULL, &for_kernel);
 
    // Transferring p and q to their respective memory buffers on the device for multiplication - similar to cudaMemcpyHostToDevice on CUDA
    for_kernel = clEnqueueWriteBuffer(command_queue, p_gpu, CL_TRUE, 0, N * sizeof(double), p, 0, NULL, NULL);
    for_kernel = clEnqueueWriteBuffer(command_queue, q_gpu, CL_TRUE, 0, N * sizeof(double), q, 0, NULL, NULL);
 
    // Creating an OpenCL program from the opencl_kernel source
    cl_program opencl_program = clCreateProgramWithSource(opencl_context, 1, (const char **)&cl_source, (const size_t *)&source_size, &for_kernel);
 
    // Building the OpenCL program
    for_kernel = clBuildProgram(opencl_program, 1, &device_id, NULL, NULL, NULL);
 
    // Creating the OpenCL kernel
    cl_kernel opencl_kernel = clCreateKernel(opencl_program, "multiply", &for_kernel);
 
    // Arguments of the OpenCL kernel for the device
    for_kernel = clSetKernelArg(opencl_kernel, 0, sizeof(cl_mem), (void *)&p_gpu);
    for_kernel = clSetKernelArg(opencl_kernel, 1, sizeof(cl_mem), (void *)&q_gpu);
    for_kernel = clSetKernelArg(opencl_kernel, 2, sizeof(cl_mem), (void *)&r_gpu);
 
    // Allocation of work items and groups - work items are similar to threads and groups are similar to blocks as on CUDA.
    size_t global_item_size = N; // Setting the global item size - similar to maximum number of threads' usage
    size_t local_item_size = 1024; // Dividing work items into groups of 1024

    // C++ and OpenCL allocations for displaying device name
    int pf_index, dev_index;
    char* device_name;
    size_t nameSize;
    cl_uint platform_count;
    cl_platform_id* platforms;
    cl_uint device_count;
    cl_device_id* devices;

    // Fetching all platforms to display device name
    clGetPlatformIDs(0, NULL, &platform_count);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platform_count);
    clGetPlatformIDs(platform_count, platforms, NULL);

    for (pf_index = 0; pf_index < platform_count; pf_index++) {

        // Fetching all OpenCL supported devices in the system
        clGetDeviceIDs(platforms[pf_index], CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * device_count);
        clGetDeviceIDs(platforms[pf_index], CL_DEVICE_TYPE_ALL, device_count, devices, NULL);

        // Display critical attributes for each device (just one here in our case)
        for (dev_index = 0; dev_index < device_count; dev_index++) {

            // Display the device name
            clGetDeviceInfo(devices[dev_index], CL_DEVICE_NAME, 0, NULL, &nameSize);
            device_name = (char*) malloc(nameSize);
            clGetDeviceInfo(devices[dev_index], CL_DEVICE_NAME, nameSize, device_name, NULL);
            printf("Device used for computation: %s\n", device_name);
            free(device_name);

        }

        free(devices);

    }

    free(platforms);
    
    cl_event event;  // Creating an event variable for timing 

    	for_kernel = clEnqueueNDRangeKernel(command_queue, opencl_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event);

    clWaitForEvents (1, &event); // Waiting for the event

    clFinish(command_queue); //Waiting until all commands have completed

    // Obtaining the start and end time for the event
    cl_ulong begin;
    cl_ulong end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(begin), &begin, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

    double duration = end - begin;

    // Printing the device computation time - note that on OpenCL, the default unit is nanoseconds in contrast to milliseconds on CUDA.
    printf("Multiplication on device computed in: %lf nanoseconds = %lf milliseconds\n", duration, duration/1000000);


    // Transferring r_gpu to its respective memory buffer on the host - similar to cudaMemcpyDeviceToHost on CUDA
    double *r = (double*)malloc(sizeof(double)*N);
    for_kernel = clEnqueueReadBuffer(command_queue, r_gpu, CL_TRUE, 0, N * sizeof(double), r, 0, NULL, NULL);
 
    // Verifying all values to be 276.0
    // fabs(q[i]-288) (absolute value) should be 0
    double maxError = 0.0;
    for (int i = 0; i < N-1; ++i){
        maxError = fmax(maxError, fabs(r[i]-276.0));
    }
    std::cout<<"\nMax error: "<<maxError<<std::endl;

    // Cleaning up memory allocations
    for_kernel = clFlush(command_queue);
    for_kernel = clFinish(command_queue);
    for_kernel = clReleaseKernel(opencl_kernel);
    for_kernel = clReleaseProgram(opencl_program);
    for_kernel = clReleaseCommandQueue(command_queue);
    for_kernel = clReleaseContext(opencl_context);
    // clReleaseMemObject on OpenCL is similar to cudaFree on CUDA.
    for_kernel = clReleaseMemObject(p_gpu);
    for_kernel = clReleaseMemObject(q_gpu);
    for_kernel = clReleaseMemObject(r_gpu);

    free(p);
    free(q);
    free(r);
    return 0;
}
