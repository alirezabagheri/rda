/* Header for Mulitnomial Logistic Regression using Regularized Dual Averaging (RDA) 
 * Alireza Bagheri Garakani (me@alirezabagheri.com), 2/20/2016
 * license: GNU GPLv3
 */

#ifndef RDA_H
#define RDA_H

#include "common_io.h"
#include "gpu.h"
#include "softmax.h"
#include <stdio.h>

 // default values
#define CUSHION_SIZE  (10 * 1024 * 1024)  // 10 MB
#define RANDOM_SEED 1

 /* Container of RDA options and associated functions to create and destroy.
 */
typedef struct {
	float lambda, gamma, eval_freq;
	size_t num_labels, max_epochs, size_minibatch, eval_max_acc_unimproved;
	matrix *y, *y_h;
	matrix_s *X, *X_h; // streaming matrix

} rda_options;

rda_options *get_rda_options() {
	// Alloc
	rda_options *res = (rda_options *)malloc(sizeof(rda_options));

	// Set defaults
	res->X = NULL;
	res->y = NULL;
	res->X_h = NULL;
	res->y_h = NULL;
	res->num_labels = 0;
	res->max_epochs = 100;
	res->size_minibatch = 50;
	res->eval_max_acc_unimproved = 10;
	res->eval_freq = 1;
	res->lambda = 0.001;
	res->gamma = 1;
	return res;
}
void free_rda_options(rda_options *opt) {
	if (opt != NULL) {
		if (opt->X != NULL) matrix_free(opt->X);
		if (opt->y != NULL) matrix_free(opt->y);
		if (opt->X_h != NULL) matrix_free(opt->X_h);
		if (opt->y_h != NULL) matrix_free(opt->y_h);
		free(opt);
	}
}

/* Kernel definition to subtract 1 from index of correct label for each example
   in predication matrix.
   INPUT:
	   * _T - device memory for prediction matrix, (rows are examples, cols are class probs)
	   * _label - device memory for (ground-truth) label vector
	   * num_labels - count of labels (i.e., cols of _T)

   OUTPUT: in-place update of device prediction matrix, _T (see description above)
*/
__global__ void _subtract_1(float *_T, const float *_label, size_t num_labels) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int row = idx / num_labels;
	int col = idx % num_labels;

	if (_label[row] == col) _T[idx]--;
}


/* Kernel definition to calculate accuracy given matrix of pre-softmax values. 
   Input is updated in-place such that, for a given row/example, if the 
   predicted class (i.e., col with largest value) matches correct class, then 
   set first element of row to 1; else 0. Thereafter, the sum of first column 
   divided by the number of rows would equal accuracy.
	INPUT:
		* _T - device memory for prediction matrix, (rows are examples, cols are class probs) 
		* _label - device memory for (ground-truth) label vector
		* num_examples - count of examples (i.e., rows of _T, or length of _label)
		* num_labels - count of labels (i.e., cols of _T)

	OUTPUT: in-place update of device prediction matrix, _T (see description above)
*/
__global__ void _eval(float *_T, const float *_label, size_t num_examples, size_t num_labels) {
	// Determine row
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= num_examples) return;

	// Offset input to point at start of the appropriate example (row).
	_T += row * num_labels;
	_label += row;

	// Find index of largest pre-softmax value
	float max_value = _T[0];
	size_t max_idx = 0; // i.e. predicted class
	for (size_t i = 1; i < num_labels; i++) {
		if (_T[i] > max_value) {
			max_value = _T[i];
			max_idx = i;
		}
	}

	// Set first element of row to 1 if predicted label is correct.
	_T[0] = (max_idx == *_label) ? 1 : 0;
}


/* Kernel definition for updating weight matrix, which is invoked on each 
   itertion of algorithm.
	INPUT:
		* _W - device memory for weight matrix
		* _W_rows - count of rows in weight matrix
		* _W_cols - count of columns in weight matrix
		* _G - device memory for gradient matrix
		* _Gnorm - device memory for group-norm vector
		* lambda - group-penalty parameter
		* scale - element-wise scaling constant

	OUTPUT: in-place update of device weight matrix argument, _W
*/
__global__ void _update_weights(float *_W, size_t _W_rows, size_t _W_cols, const float *_G, const float *_Gnorm, float lambda, float scale) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < 0 || idx >= _W_rows * _W_cols) return;
	int row = idx / _W_cols;

	_W[idx] = scale * _G[idx];

	if (row != 0) {
		if (_Gnorm[row] > lambda) _W[idx] *= 1 - (lambda / _Gnorm[row]);
		else _W[idx] = 0;
	}
}

#endif
