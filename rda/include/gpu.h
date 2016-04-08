/* General GPU kernel functions
 * Alireza Bagheri Garakani (me@alirezabagheri.com), 2/20/2016
 * license: GNU GPLv3
 */

#ifndef GPU_H
#define GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

// Global constants
#define THREADS_PER_BLOCK 256

/* Kernel defintions to be called using GPU_LAUNCH wrapper
*/

#define GPU_LAUNCH(kernel_name, num_blocks, ...)     \
    kernel_name <<< (num_blocks), THREADS_PER_BLOCK >>> (__VA_ARGS__); \
    cudaDeviceSynchronize();

// Used as breakpint for debugging device memory
__global__ void _mon(float *input) {
	return;
}

// Apply element-wise cosine operation (IN-PLACE)
__global__ void _cosine(float *input, size_t N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 0 && idx < N) input[idx] = __cosf(input[idx]);
}

// Apply element-wise square operation
__global__ void _square(float *output, const float *input, size_t N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 0 && idx < N) output[idx] = input[idx] * input[idx];
}

// Apply element-wise sqrt operation  (IN-PLACE)
__global__ void _sqrt(float *input, size_t N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 0 && idx < N) input[idx] = __powf(input[idx], 0.5F);
}

// Set all elements of vector to value
__global__ void _set(float *input, size_t N, float value) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 0 && idx < N) input[idx] = value;
}

// Set to 1 if non-zero value  (IN-PLACE)
__global__ void _set_nonzero(float *input, size_t N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 0 && idx < N && input[idx] != 0) input[idx] = 1;
}


/* Function is wrapper around kernel function that initializes random number generator
	and sets each element of input to uniform random value in [+1,-1]  (IN-PLACE)
		INPUT:
			* input - matrix with N elements
			* N - number of elements in matrix/vector input
			* seed - random seed unsigned integer

		OUTPUT:
			* on success, initializes each element of _input to uniform random
				value in [+1,-1] and returns cudaError_t representing success. 
				Otherwise, returns error as cudaError_t.
*/

__global__ void _uniform_rand(float *input, curandState_t* states, size_t N, size_t seed);

cudaError_t set_uniform_rand(float *_input, size_t N, size_t seed){
	cudaError_t error;

	// Allocate device memory for random state object for each N elements
	curandState_t* _states = NULL;
	error = cudaMalloc((void**)&_states, sizeof(curandState_t) * N);
	if (error != cudaSuccess) {
		printf("cudaMalloc failed for random state vector!");
		return error;
	}

	// Generate random numbers on GPU
	_uniform_rand << < (size_t)(N / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (_input, _states, N, seed);
	cudaDeviceSynchronize();
	if ((error = cudaGetLastError()) != cudaSuccess) {
		printf("Failed to generate random number on GPU (error code %s)!\n", cudaGetErrorString(error));
		return error;
	}

	// No longer need random state memory
	cudaFree(_states);
	
	return cudaSuccess;
}

__global__ void _uniform_rand(float *input, curandState_t* states, size_t N, size_t seed) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 0 && idx < N) {
		curand_init(seed, idx, 0, &states[idx]);
		input[idx] = (curand_uniform(&states[idx]) * 2) - 1;
	}
}



// GPU Matrix struct and constructing function to use as shorthand for creating GPU matrices
typedef struct {
	size_t row, col, elem, num_blocks;
	float *data;
} gpu_matrix;

gpu_matrix *gpu_matrix_create(size_t row, size_t col, bool init, float init_value) {
	cudaError_t error;

	// allocate host mem for struct container
	gpu_matrix *mat = NULL;
	if ((mat = (gpu_matrix *)malloc(sizeof(gpu_matrix))) == NULL) {
		printf("Failed to allocate memory for GPU matrix container struct.\n");
		return NULL;
	}
	mat->row = row;
	mat->col = col;
	mat->elem = row * col;
	mat->num_blocks = (size_t)(mat->elem / THREADS_PER_BLOCK) + 1;


	// allocate device memory for matrix
	mat->data = NULL;
	error = cudaMalloc((void**)&(mat->data), sizeof(float) * mat->elem);
	if (error != cudaSuccess) {
		printf("cudaMalloc failure!\n");
		free(mat);
		return NULL;
	}

	// initialize, if enabled
	if (init) {
		GPU_LAUNCH(_set, mat->num_blocks, mat->data, mat->elem, init_value);
		if ((error = cudaGetLastError()) != cudaSuccess) {
			printf("Failed to initialize values for matrix (error code %s)!\n", cudaGetErrorString(error));
			cudaFree(mat->data);
			free(mat);
			return NULL;
		}
	}

	// return struct
	return mat;
}

gpu_matrix *gpu_matrix_create(size_t row, size_t col, bool init) {
	return gpu_matrix_create(row, col, init, 0);
}

#endif
