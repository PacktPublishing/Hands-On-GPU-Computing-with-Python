
#include <hip/hip_runtime.h>
#include <iostream>

#define N 500000000 //500 Million Elements
#define THREADS_PER_BLOCK 1024

// GPU kernel function to multiply two array elements and also update the results on the second array
__global__ void multiply(double *p, double *q, unsigned long n){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n)
		q[index] = p[index] * q[index];
}


int main(void) {

	double *p, *q; // host copies of p, q
	//double *gpu_p, *gpu_q; // device copies of p, q
	unsigned long size = N * sizeof(unsigned long); // we need space for N unsigned long integers
	unsigned long i;
	
	/*	
	// Allocate GPU/device copies of gpu_p, gpu_q
	cudaMalloc((void**)&gpu_p, size);
	cudaMalloc((void**)&gpu_q, size);


	// Allocate CPU/host copies of p, q
	p = (double *)malloc(size);
	q = (double *)malloc(size);
	*/

	//Unified Memory Allocation for CPU and GPU
 	cudaMallocManaged((void**)&p, size);
  	cudaMallocManaged((void**)&q, size);


	// Setup input values
	for (i = 0; i < N - 1; ++i)
	{
		p[i] = 24.0;
		q[i] = 12.0;
	}

	/*
	// Copy inputs to device
	cudaMemcpy(gpu_p, p, size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_q, q, size, cudaMemcpyHostToDevice);
	*/

	//INITIALIZE CUDA EVENTS
	hipEvent_t start, stop;
	float elapsedTime; 

	//CREATING EVENTS
	hipEventCreate(&start);
	hipEventCreate(&stop);
	hipEventRecord(start, 0);
	cudaMemPrefetchAsync(p, N * sizeof(double), 0);
	cudaMemPrefetchAsync(q, N * sizeof(double), 0);
	hipDeviceSynchronize();

	//CUDA KERNEL STUFF HERE...
	// Launch multiply() kernel on GPU with N threads
	hipLaunchKernelGGL(multiply, dim3((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0, p, q, N);

	//FINISH RECORDING
	hipEventRecord(stop, 0);
	hipDeviceSynchronize();
	hipEventSynchronize(stop);

	//CALCULATE ELAPSED TIME
	hipEventElapsedTime(&elapsedTime, start, stop);

	//DISPLAY COMPUTATION TIME
	
	hipDeviceProp_t prop;
	int count;

	hipGetDeviceCount(&count);

	for (int igtx = 0; igtx < count; igtx++) {
		hipGetDeviceProperties(&prop, igtx);
		printf("\nGPU Device used for computation: %s\n", prop.name);
		printf("\nMultiplication on GPU computed in: %f milliseconds", elapsedTime);
	}

	
	/*
	// Copy device result back to host copy of q
	cudaMemcpy(q, gpu_q, size, cudaMemcpyDeviceToHost);
	*/
	
  	// Verifying all values to be 288.0
	// fabs(q[i]-288) (absolute value) should be 0
	double maxError = 0.0;
	for (int i = 0; i < N-1; ++i){
    		maxError = fmax(maxError, fabs(q[i]-288.0));
	}
  	std::cout << "\nMax error: " << maxError << std::endl;

	// Clean unified memory allocations
	hipFree(p);
	hipFree(q);

	return 0;
}
