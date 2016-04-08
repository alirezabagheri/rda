/* Softmax function for GPU
 * Alireza Bagheri Garakani (me@alirezabagheri.com), 2/20/2016
 * license: GNU GPLv3
 */

#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "gpu.h"
#include "float.h"

#define SOFTMAX_MIN FLT_MIN
#define SOFTMAX_MAX FLT_MAX

/* [DEPRECATED]
    Performs softmax using single kernel call and no auxilary storage. Limitation here is that
	num_labels must be <= hardware's max-threads-per-block. If the number of classes is too large
	for this method, see Method 2.
	Adapted from 'torch/cunn' (https://github.com/torch/cunn/blob/master/SoftMax.cu)
*/ 
//__global__ void _softmax_single_kernel(float *input, size_t num_examples, size_t num_labels){
//	__shared__ float s_val;
//
//	// Return if out of range.
//	if (threadIdx.x >= num_labels || blockIdx.x >= num_examples) return;
//
//	// Offset input vector to point at start of the
//	// appropriate example (row) as indicated block id.
//	input += blockIdx.x * num_labels;
//
//
//	// Find largest pre-softmax value (s_val is MAX)
//	if (threadIdx.x == 0){
//		s_val = input[0];
//		for (size_t i = 1; i < num_labels; i++){
//			if (input[i] > s_val) s_val = input[i];
//		}
//	}
//
//	__syncthreads();
//
//	// compute unnormalized softmax values
//	input[threadIdx.x] = __expf(input[threadIdx.x] - s_val);
//	if (input[threadIdx.x] < SOFTMAX_MIN) input[threadIdx.x] = 0;
//
//	__syncthreads();
//
//	// Compute sum of un-norm values  (s_val is SUM)
//	if (threadIdx.x == 0){
//		s_val = 0;
//		for (size_t i = 0; i < num_labels; i++) s_val += input[i];
//	}
//	__syncthreads();
//
//	// normalize
//	if (s_val != 0) input[threadIdx.x] = input[threadIdx.x] / s_val;
//}



/* Perform softmax operation using four kernel calls.
	INPUT:
		* _input - matrix of size num_examples x num_labels (in row major vector representation)
		* _aux - auxilary storage vector of length num_examples.
		* _ones - ones vector of length num_labels
		* num_examples  - number of examples in matrix _input
		* num_labels - number of labels/classes
		* handle - handle to cublas

	OUTPUT:
		* on success, performs softmax operation on each row of _input in-place and returns
			cudaError_t representing success. Otherwise, return error as cudaError_t.
*/

__global__ void _softmax_kernel_1(float *input, float *aux, size_t num_examples, size_t num_labels);
__global__ void _softmax_kernel_2(float *input, float *aux, size_t num_examples, size_t num_labels);
__global__ void _softmax_kernel_3(float *input, float *aux, size_t num_examples, size_t num_labels);

cudaError_t softmax(float *_input, float *_aux, float *_ones, size_t num_examples, size_t num_labels, cublasHandle_t handle) {
	cublasStatus_t status;
	cudaError_t error;
	const float one = 1, zero = 0;

	if (_input == NULL || _aux == NULL) return cudaErrorLaunchFailure;

	// Determine number of blocks to set
	const size_t num_blocks_elem = (size_t)round((num_examples * num_labels / THREADS_PER_BLOCK) + 0.5);
	const size_t num_blocks_row = (size_t)round((num_examples / THREADS_PER_BLOCK) + 0.5);

	// [1] For each row, find max softmax value (i.e. example) and store in aux vector
	_softmax_kernel_1 << < num_blocks_row, THREADS_PER_BLOCK >> > (_input, _aux, num_examples, num_labels);
	cudaDeviceSynchronize();
	if ((error = cudaGetLastError()) != cudaSuccess) {
		printf("Failed to launch _softmax_kernel_1 (error code %s)!\n", cudaGetErrorString(error));
		return error;
	}

	// [2] For each elem, subtract max from aux vector and apply exp() function.
	_softmax_kernel_2 << < num_blocks_elem, THREADS_PER_BLOCK >> > (_input, _aux, num_examples, num_labels);
	cudaDeviceSynchronize();
	if ((error = cudaGetLastError()) != cudaSuccess) {
		printf("Failed to launch _softmax_kernel_2 (error code %s)!\n", cudaGetErrorString(error));
		return error;
	}

	// [3] For each row, sum and store result in aux vector; use cuBLAS.
	status = cublasSgemv(handle, CUBLAS_OP_T, num_labels, num_examples, &one, _input, num_labels, _ones, 1, &zero, _aux, 1);
	cudaDeviceSynchronize();
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("cublasSgemv, as part of softmax function, returned error code %d !\n", error);
		return cudaErrorLaunchFailure;
	}

	// [4] For each elem, divide by sum to normalize (i.e. produce valid probability)
	_softmax_kernel_3 << < num_blocks_elem, THREADS_PER_BLOCK >> > (_input, _aux, num_examples, num_labels);
	cudaDeviceSynchronize();
	if ((error = cudaGetLastError()) != cudaSuccess) {
		printf("Failed to launch _softmax_kernel_3 (error code %s)!\n", cudaGetErrorString(error));
		return error;
	}

	return cudaSuccess;
}


/* (Helper function for softmax function)
	For each row, find max softmax value (i.e. example) and store in aux vector
*/
__global__ void _softmax_kernel_1(float *input, float *aux, size_t num_examples, size_t num_labels) {
	// Determine row
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= num_examples) return;

	// Offset input to point at start of the appropriate example (row).
	input += row * num_labels;
	aux += row;

	// Determine max value for row
	float max_value = input[0];
	for (size_t i = 1; i < num_labels; i++) {
		if (input[i] > max_value) max_value = input[i];
	}

	// Set in aux variable
	*aux = max_value;
}

/* (Helper function for softmax function)
	For each elem, subtract max from aux vector and apply exp() function.
*/
__global__ void _softmax_kernel_2(float *input, float *aux, size_t num_examples, size_t num_labels) {
	// Determine elem
	int elem = blockIdx.x * blockDim.x + threadIdx.x;
	if (elem >= num_examples * num_labels) return;

	// Determine row
	int row = elem / num_labels;
	if (row >= num_examples) return;

	// Subtract max row softmax value and apply exp() function
	input[elem] = __expf(input[elem] - aux[row]);
	if (input[elem] < SOFTMAX_MIN) input[elem] = 0;
}

/* (Helper function for softmax function)
	For each elem, divide by sum to normalize (i.e. produce valid probability)
*/
__global__ void _softmax_kernel_3(float *input, float *aux, size_t num_examples, size_t num_labels) {
	// Determine elem
	int elem = blockIdx.x * blockDim.x + threadIdx.x;
	if (elem >= num_examples * num_labels) return;

	// Determine row
	int row = elem / num_labels;
	if (row >= num_examples) return;

	// Normalize
	if (aux[row] != 0) input[elem] = input[elem] / aux[row];
}


#endif