/* Mulitnomial Logistic Regression using Regularized Dual Averaging (RDA)
 * Stochastic, Group-Lasso Regularizaed 
 * Alireza Bagheri Garakani (me@alirezabagheri.com), 2/20/2016
 * license: GNU GPLv3
 *
 * NOTES:
 *  - Currently, we fit largest heldout chunk and single train minibatch into
 *    device mem. This is geared towards more frequent heldout evaluations, also
 *    generally more appropriate for online learning. Alternative solution is to
 *    use same chunksize for both datasets: allows faster epoch time and re-use 
 *    of certaindevice variables (e.g, _T and _X)
 *
 * TODO:
 *  - better input reading and validation
 *  - better error handling and clean-up
 */

#include "RDA.h"


/* Implements mulitnomial logistic regression using regularized dual averaging (RDA)

 	Input:
		* devID - int, GPU device id
		* opt - rda_options struct, options for RDA

	Output:
		* on success, returns point to matrix struct of learned parameter weights.
			otherwise, returns NULL.
	
	Notes:
		* weights corresponding to best heldout evaluation are selected. If learning ends
			prior to any heldout evaluation, then last iteration weights are returned.
*/
matrix *RDA(int devID, const rda_options* opt){

	// Check arguments
	if (opt == NULL) {
		printf("must provide non-NULL rda_options struct\n");
		return NULL;
	}

	// Create reference variables for brevity
	matrix_s *X = opt->X, *X_h = opt->X_h;
	matrix *y = opt->y, *y_h = opt->y_h;
	const size_t num_covariates = X->col + 1; // include 1 for intercept term

	// Prepare result matrix (matrix is updated within algo loop) 
	matrix *W = (matrix *)malloc(sizeof(matrix));
	W->row = num_covariates;
	W->col = opt->num_labels;
	W->data = NULL;

	// Define variables to read CUDA status/error
	cudaError_t error;
	cublasStatus_t status;

	// Set device
	if ((error = cudaSetDevice(devID)) != cudaSuccess){
		printf("cudaSetDevice returned error code %d", error);
		return NULL;
	}

	// Create CUBLAS handle and necessary constants
	const float one = 1, zero = 0;
	cublasHandle_t handle;
	if ((status = cublasCreate(&handle)) != CUBLAS_STATUS_SUCCESS){
		printf("CUBLAS initialization failed\n");
		return NULL;
	}

	//// MEMORY ALLOCATION

	// DEVICE memory for weight matrix W (covariates x num_labels).
	// Initialize to value 0
	gpu_matrix *_W = gpu_matrix_create(num_covariates, opt->num_labels, true);
	if (_W == NULL) {
		printf("failed to allocate device memory for weight matrix!");
		return NULL;
	}

	// DEVICE memory for (cumulative) gradient matrix G (covariates x num_labels)
	// Initialize to value 0
	gpu_matrix *_G = gpu_matrix_create(num_covariates, opt->num_labels, true);
	if (_G == NULL) {
		printf("failed to allocate device memory for gradient matrix!");
		return NULL;
	}


	// DEVICE memory for G_temp, used to compute group norm of covariates. (covariates x num_labels)
	gpu_matrix *_G_temp = gpu_matrix_create(num_covariates, opt->num_labels, false);
	if (_G_temp == NULL) {
		printf("failed to allocate device memory for temporary gradient matrix!");
		return NULL;
	}

	// DEVICE memory for group-norm vector. (1 x covariates)
	gpu_matrix *_g = gpu_matrix_create(1, num_covariates, false);
	if (_g == NULL) {
		printf("failed to allocate device memory for group-norm vector!");
		return NULL;
	}

	// DEVICE memory for prediction matrix T. (minibatch_size x num_labels)
	gpu_matrix *_T = gpu_matrix_create(opt->size_minibatch, opt->num_labels, false);
	if (_T == NULL) {
		printf("failed to allocate device memory for prediction matrix!");
		return NULL;
	}

	// DEVICE memory for (mini-batch) data matrix X. (num_covariates x minibatch_size)
	// Initialize to value 1 (since we require intercept be 1)
	gpu_matrix *_X = gpu_matrix_create(opt->size_minibatch, num_covariates, true, 1.0);
	if (_X == NULL) {
		printf("failed to allocate device memory for data matrix!");
		return NULL;
	}

	// DEVICE memory for (mini-batch) label vector. (1 x minibatch_size)
	gpu_matrix *_y = gpu_matrix_create(1, opt->size_minibatch, false);
	if (_y == NULL) {
		printf("failed to allocate device memory for label vector!");
		return NULL;
	}
	

	// HOST memory for (mini-batch) label vector. Note: We cannot directly use 'opt->y' variable
	// since randomization will not allow us to have contiguous memory for device copy.
	float *y_aligned = NULL;
	if ((y_aligned = (float *) malloc(sizeof(float)*opt->size_minibatch)) == NULL){
		printf("host malloc failed for label vector!");
		return NULL;
	}

	// DEVICE memory for ones vector. (1 x num_labels)
	// Initialize to value 1
	gpu_matrix *_ones = gpu_matrix_create(1, opt->num_labels, true, 1.0);
	if (_ones == NULL) {
		printf("failed to allocate device memory for ones vector!");
		return NULL;
	}

	// DEVICE memory for heldout label vector. (init from host now)
	gpu_matrix *_y_h = gpu_matrix_create(1, X_h->row_total, false);
	if (_y_h == NULL) {
		printf("failed to allocate device memory for heldout label vector!");
		return NULL;
	}
	error = cudaMemcpy(_y_h->data, y_h->data, sizeof(float)* X_h->row_total, cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
		printf("Failed to initalize heldout label vector (error code %s)!\n", cudaGetErrorString(error));
		return NULL;
	}

	// Determine largest size of heldout samples that can be loaded into memory at once. This allows
	// more efficient evaluation on heldout.
	size_t mem_free, mem_total;
	error = cudaMemGetInfo(&mem_free, &mem_total);
	if (error != cudaSuccess) {
		printf("Failed to check available memory on device (error code %s)!\n", cudaGetErrorString(error));
		return NULL;
	}
	size_t eval_chunk_size = (mem_free - CUSHION_SIZE) / (sizeof(float) * (num_covariates + opt->num_labels));
	if (eval_chunk_size == 0) {
		printf("Cannot fit even single heldout instance into device mem (error code %s)!\n", cudaGetErrorString(error));
		return NULL;
	}
	size_t eval_partitions = (X_h->row_total / eval_chunk_size) + (X_h->row_total % eval_chunk_size == 0 ? 0 : 1);


	// DEVICE memory for heldout data matrix X_h. (chunk_size x num_covariates)
	// Initialize to value 1 (since we require intercept be 1)
	gpu_matrix *_X_h = gpu_matrix_create(eval_chunk_size, num_covariates, true, 1.0);
	if (_X_h == NULL) {
		printf("failed to allocate device memory for heldout data matrix!");
		return NULL;
	}

	// DEVICE memory for intermediate matrix T_h. (chunk_size x num_labels)
	gpu_matrix *_T_h = gpu_matrix_create(eval_chunk_size, opt->num_labels, false);
	if (_T_h == NULL) {
		printf("failed to allocate device memory for heldout prediction matrix!");
		return NULL;
	}
	
	// If entire heldout fits in memory, move data in once and for all.
	if (eval_partitions == 1) {

		// Read X_h into host memory
		matrix_stream_read(X_h, X_h->row_total);
		if (X_h->row != X_h->row_total) {
			printf("Failed to read entire heldout set into host memory!\n");
			return NULL;
		}

		// Copy X_h into device memory, leaving first column as 1 for intecept term
		for (size_t i = 0; i < X_h->row; i++) {
			error = cudaMemcpy(_X_h->data + 1 + (num_covariates * i), X_h->data + (X_h->col * i), sizeof(float)* X_h->col, cudaMemcpyHostToDevice);
			if (error != cudaSuccess) {
				printf("Failed to call cudaMemcpy for heldout data (error code %s)!\n", cudaGetErrorString(error));
				return NULL;
			}
		}

	} else {
		printf("Note: heldout set must be loaded into device with chunk size = %zu (%zu parts).\n",
			eval_chunk_size, eval_partitions);
	}


	//// BEGIN ALGORITHM

	printf("Running algorithm.\n");

	size_t t = 0, eval_count_acc_unimproved = 0;
	float scale, scale2, acc_h = 0, acc_h_best = 0, acc_h_part, count_nonzero_covariates;

	size_t num_iter_per_eval = (size_t)(((float)X->row_total * opt->eval_freq) / ((float)opt->size_minibatch));
	if (num_iter_per_eval == 0) num_iter_per_eval = 1;

	size_t num_iter_per_print = num_iter_per_eval / 10;
	if (num_iter_per_print == 0) num_iter_per_print = 1;

	while (matrix_stream_read(X, opt->size_minibatch)){
		t++; // increment iteration count

		// Sanity check
		if (X->row != opt->size_minibatch){
			printf("rows returned not equal to mini-batch size; unexpected condition.\n");
			break;
		}

		// [0.1] Read data matrix into device, leaving first column as 1 for intecept term
		for (size_t i = 0; i < opt->size_minibatch; i++) {
			error = cudaMemcpy(_X->data + 1 + (num_covariates * i), X->data + (X->col * i), sizeof(float) * X->col, cudaMemcpyHostToDevice);
			if (error != cudaSuccess) {
				printf("Failed to set data matrix (error code %s)!\n", cudaGetErrorString(error));
				return NULL;
			}
		}

		// [0.2] Read label vector into device. Since our samples are randomized, ensure labels
		//		 are properly aligned to respective samples.
		for (size_t i = 0; i < opt->size_minibatch; i++) y_aligned[i] = y->data[X->row_idx[i]];
		error = cudaMemcpy(_y->data, y_aligned, sizeof(float) * opt->size_minibatch, cudaMemcpyHostToDevice);
		if (error != cudaSuccess){
			printf("Failed to set label vector (error code %s)!\n", cudaGetErrorString(error));
			return NULL;
		}
		
		// [1] Compute gradient of loss function
		// [1.1] Perform W * P
		status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, opt->num_labels, opt->size_minibatch,
				num_covariates, &one, _W->data, _W->col, _X->data, _X->col, &zero, _T->data, _T->col);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS){
			printf("cublasSgemv, as part of grad comp, returned error code %d!\n", error);
			return NULL;
		}

		// [1.2] Apply softmax. Note: use of _X on current iteration is complete, thus we can 
		// safely provided it below as auxilary storage for softmax function (gaurentee to be
		// large enough b/c aux must have enough space for vector of length minibatch size).
		error = softmax(_T->data, _X->data, _ones->data, opt->size_minibatch, opt->num_labels, handle);
		if (error != cudaSuccess){
			printf("Failed to launch kernel (error code %s)!\n", cudaGetErrorString(error));
			return NULL;
		}

		// [1.3] Subtract 1 from vector at true label
		GPU_LAUNCH(_subtract_1, _T->num_blocks, _T->data, _y->data, opt->num_labels);
		cudaDeviceSynchronize();
		if ((error = cudaGetLastError()) != cudaSuccess){
			printf("Failed to launch kernel _subtract_1 (error code %s)!\n", cudaGetErrorString(error));
			return NULL;
		}

		// [1.4] Update average gradient using product of _G_t and _X.
		scale = 1.0f / t; // Used for new gradient
		scale2 = 1 - scale; // Used for previous gradients
		status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, opt->num_labels, num_covariates,
			opt->size_minibatch, &scale, _T->data, _T->col, _X->data, _X->col, &scale2, _G->data, _G->col);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS){
			printf("cublasSgemv, as part of gradient update, returned error code %d!\n", error);
			return NULL;
		}


		// [2] Compute group norm.
		// [2.1] Square all elements (and put result into temporary matrix G_temp)
		GPU_LAUNCH(_square, _G->num_blocks, _G_temp->data, _G->data, _G->elem);
		if ((error = cudaGetLastError()) != cudaSuccess){
			printf("Failed to launch kernel _square (error code %s)!\n", cudaGetErrorString(error));
			return NULL;
		}
		// [2.2] Row-wise summation
		status = cublasSgemv(handle, CUBLAS_OP_T, opt->num_labels, num_covariates, &one, _G_temp->data, _G_temp->col, _ones->data, 1, &zero, _g->data, 1);
		cudaDeviceSynchronize();
		if (status != CUBLAS_STATUS_SUCCESS){
			printf("cublasSgemv, as part of row-wise summation, returned error code %d!\n", error);
			return NULL;
		}
		// [2.3] Sqrt all elements of vector to get proper group-norm value
		GPU_LAUNCH(_sqrt, _g->num_blocks, _g->data, _g->elem);
		if ((error = cudaGetLastError()) != cudaSuccess){
			printf("Failed to launch kernel _sqrt (error code %s)!\n", cudaGetErrorString(error));
			return NULL;
		}


		// [3] Update weight matrix. (intercept update is special; see kernel function definition)
		scale = -powf(t, 0.5F) / (opt->size_minibatch * opt->gamma);
		GPU_LAUNCH(_update_weights, _W->num_blocks, _W->data, _W->row, _W->col, _G->data, _g->data, opt->lambda, scale);
		if ((error = cudaGetLastError()) != cudaSuccess){
			printf("Failed to launch kernel _update_weights (error code %s)!\n", cudaGetErrorString(error));
			return NULL;
		}

		// [4] Heldout Evaluation / Stopping Criteria

		if (t % num_iter_per_print == 0) fprintf(stderr, "*");

		if (t % num_iter_per_eval == 0 || X->num_loops >= opt->max_epochs) {

			// [H] Compute heldout accuracy
			acc_h = 0;
			for (size_t p = 0; p < eval_partitions; p++){

				// [H.0] Unless the entire heldout data is on device (eval_partitions = 1), we need prepare _X_h matrix.
				if (eval_partitions != 1){

					// Read part of X_h into host memory
					if (p == eval_partitions - 1 && X_h->row_total % eval_chunk_size != 0) { 
						// last partition is not full; read only whatever is left.
						matrix_stream_read(X_h, X_h->row_total % eval_chunk_size);
						if (X_h->row != X_h->row_total % eval_chunk_size){
							printf("Failed to read heldout set part into host memory!\n");
							return NULL;
						}
					} else {
						matrix_stream_read(X_h, eval_chunk_size);
						if (X_h->row != eval_chunk_size){
							printf("Failed to read heldout set part into host memory!\n");
							return NULL;
						}
					}
					
					// Read into temporary memory for heldout data matrix X into memory, leaving first column as 1 for intecept term
					for (size_t i = 0; i < X_h->row; i++) {
						error = cudaMemcpy(_X_h->data + 1 + (num_covariates * i), X_h->data + (X_h->col * i), sizeof(float)* X_h->col, cudaMemcpyHostToDevice);
						if (error != cudaSuccess) {
							printf("Failed to call cudaMemcpy for heldout data (error code %s)!\n", cudaGetErrorString(error));
							return NULL;
						}
					}
					
				}


				// [H.1] Perform W * P
				status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, opt->num_labels, X_h->row,
					num_covariates, &one, _W->data, _W->col, _X_h->data, _X_h->col, &zero, _T_h->data, _T_h->col);
				cudaDeviceSynchronize();
				if (status != CUBLAS_STATUS_SUCCESS){
					printf("cublasSgemv, as part of heldout acc, returned error code %d after launching!\n", status);
					return NULL;
				}

				// [H.2] For each example/row, set first column to 1 if correct prediction.
				GPU_LAUNCH(_eval, _T_h->num_blocks, _T_h->data, _y_h->data + (p * eval_chunk_size), X_h->row, opt->num_labels);
				if ((error = cudaGetLastError()) != cudaSuccess){
					printf("Failed to launch kernel _eval (error code %s)!\n", cudaGetErrorString(error));
					return NULL;
				}

				// [H.3] Sum first column of matrix to get correctly predicted count. (see _eval kernel definition)
				acc_h_part = 0;
				status = cublasSasum(handle, X_h->row, _T_h->data, opt->num_labels, &acc_h_part);
				cudaDeviceSynchronize();
				if (status != CUBLAS_STATUS_SUCCESS){
					printf("cublasSasum, as part of heldout acc, returned error code %d!\n", status);
					return NULL;
				}
				acc_h += acc_h_part;

			}
			// Average to get accuracy
			acc_h /= X_h->row_total;

			// [H.4] Check number of non-zero covariates (use variable _g as temp)
			// first, row-wise summation
			status = cublasSgemv(handle, CUBLAS_OP_T, opt->num_labels, num_covariates, &one, _W->data, _W->col, _ones->data, 1, &zero, _g->data, 1);
			cudaDeviceSynchronize();
			if (status != CUBLAS_STATUS_SUCCESS) {
				printf("cublasSgemv, as part of row-wise summation, returned error code %d!\n", error);
				return NULL;
			}
			// then, convert non-zero entries to 1
			GPU_LAUNCH(_set_nonzero, _g->num_blocks, _g->data, _g->elem);
			if ((error = cudaGetLastError()) != cudaSuccess) {
				printf("Failed to launch kernel _sqrt (error code %s)!\n", cudaGetErrorString(error));
				return NULL;
			}
			// then, sum vector
			count_nonzero_covariates = 0;
			status = cublasSasum(handle, _g->elem, _g->data, 1, &count_nonzero_covariates);
			cudaDeviceSynchronize();
			if (status != CUBLAS_STATUS_SUCCESS) {
				printf("cublasSasum, as part of heldout acc, returned error code %d!\n", status);
				return NULL;
			}
			

			// [H.5] Check convergence
			if (acc_h > acc_h_best) {
				eval_count_acc_unimproved = 0;
				acc_h_best = acc_h;

				// Update host memory with current best W
				if (W->data == NULL) W->data = (float *)malloc(sizeof(float)* W->row * W->col);
				error = cudaMemcpy(W->data, _W->data, sizeof(float)* W->row * W->col, cudaMemcpyDeviceToHost);
				if (error != cudaSuccess){
					printf("Failed to call cudaMemcpy in copying model to host (error code %s)!\n", cudaGetErrorString(error));
					return NULL;
				}
			}
			else {
				eval_count_acc_unimproved++;
			}

			// [H.6] Print 
			printf("\niteration %zu (epoch %zu),  non-zero covariates: %.0f (%.2f%%),  heldout accuracy: %.2f%% (best %.2f%%; unimproved %zu)%s\n",
				t, X->num_loops + 1, count_nonzero_covariates, count_nonzero_covariates * 100 / ((float)num_covariates), 
				acc_h * 100, acc_h_best * 100, eval_count_acc_unimproved, X->num_loops >= opt->max_epochs ? " (FINAL)" : "");

			// [H.7] Check stopping criteria
			if (eval_count_acc_unimproved >= opt->eval_max_acc_unimproved){
				printf("\nReach max consecutive eval without heldout accuracy improvement.\n");
				break;
			}
		}

		// Check whether max_epoch is reached. [Note: if 'num_examples % opt->size_minibatch != 0', then
		// some examples from the 'opt->max_epochs + 1'th epoch (and even onward epochs if
		// opt->size_minibatch >>> num_examples) would have been used on current iteration.]
		if (X->num_loops >= opt->max_epochs){
			printf("\nReach max epochs: %zu (on iteration %zu)\n", opt->max_epochs, t);
			break;
		} 

	}



	//// PREPARE RESULT

	// Copy output vector from GPU buffer to host memory, if has never been defined yet (i.e. no chance for evals).
	if (W->data == NULL){
		W->data = (float *)malloc(sizeof(float)* W->row * W->col);
		error = cudaMemcpy(W->data, _W->data, sizeof(float)* W->row * W->col, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess){
			printf("Failed to call cudaMemcpy in copying model to host (error code %s)!\n", cudaGetErrorString(error));
			return NULL;
		}
	}
	

	//// CLEAN-UP

	// Release CUBLAS handle
	if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS){
		printf("cublasDestroy failed; oh well!");
	}

	// Release Device and all resources.
	if ((error = cudaDeviceReset()) != cudaSuccess) {
		printf("cudaDeviceReset failed (error code %s); oh well!", cudaGetErrorString(error));
	}


	return W;
}


/*  MAIN
 *  Reads program arguments and intialiates RDA.
*/
int main(int argc, char *argv[]) {

	// Print usage
	if (argc != 14 && argc != 15) {
		printf("Usage: %s <device-id> <path:W> <path:X> <path:y> <path:X_h> <path:y_h> "
			"<float:lambda> <float:gamma> <size:minibatch_size> <size:max_epochs> "
			"<float:eval_freq> <size:eval_max_acc_unimproved> [<size: random_seed>]\n", argv[0]);
		printf("\t\t device-id - gpu device id (e.g., 0 or 1)\n");
		printf("\t\t W - output matrix: D x K\n");
		printf("\t\t X - data matrix N x D\n");
		printf("\t\t y - label vector (N x 1), where each element is in [0 ... K-1]\n");
		printf("\t\t X_h - heldout data matrix: M x D\n");
		printf("\t\t y_h - heldout label vector (M x 1), where each element is in [0 ... K-1]\n");
		printf("\t\t lambda - group-penalty coeff\n");
		printf("\t\t gamma - dual averaging parameter\n");
		printf("\t\t num_labels - number of unique labels K\n");
		printf("\t\t minibatch_size - size of minibatch used for each gradient step\n");
		printf("\t\t max_epochs - maximum number of epochs allowed for training\n");
		printf("\t\t eval_freq - frequency of heldout evaluation (e.g., use 1.0 for eval after each training epoch)\n");
		printf("\t\t eval_max_acc_unimproved - treshold number of evaluations before terminating due to convergence (e.g., 10)\n");
		printf("\t\t random_seed - set seed effecting permutation of training data. (optional, default: %zu)\n", (size_t)RANDOM_SEED);
		printf("\n\n");
		return 1;
	}

	// Initialize RDA options struct
	rda_options *opt = get_rda_options();

	// Read arguments
	const int devID = atoi(argv[1]);
	const char * filename_W = argv[2];
	const char * filename_X = argv[3];
	const char * filename_y = argv[4];
	const char * filename_X_h = argv[5];
	const char * filename_y_h = argv[6];
	opt->lambda = atof(argv[7]);
	opt->gamma = atof(argv[8]);
	opt->num_labels = (size_t)atoi(argv[9]);
	opt->size_minibatch = (size_t)atoi(argv[10]);
	opt->max_epochs = (size_t)atoi(argv[11]);
	opt->eval_freq = atof(argv[12]);
	opt->eval_max_acc_unimproved = (size_t)atoi(argv[13]);

	size_t random_seed = (size_t)RANDOM_SEED;
	if (argc == 15) {
		random_seed = (size_t)atoi(argv[14]);
	}
	
	// Open stream to input data (permutable, file-looping)
	if ((opt->X = matrix_stream_create(filename_X, true, true)) == NULL){
		printf("Creating data matrix stream failed!");
		return 2;
	}
	// Set random seed for training data
	set_seed(opt->X, random_seed);

	// Read input labels
	if ((opt->y = matrix_read(filename_y, false)) == NULL){
		printf("Reading label vector failed!");
		return 2;
	}
	// Open stream to heldout data (file-looping)
	if ((opt->X_h = matrix_stream_create(filename_X_h, false, true)) == NULL){
		printf("Reading heldout data matrix failed!");
		return 2;
	}
	// Read heldout labels
	if ((opt->y_h = matrix_read(filename_y_h, false)) == NULL){
		printf("Reading heldout label vector failed!");
		return 2;
	}

	// Print parsed arguments to stdout	
	printf("lambda = %g, gamma = %g, num_labels = %zu, size_minibatch = %zu, max_epochs = %zu, "
		"eval_freq = %g, eval_max_acc_unimproved = %zu, random_seed = %zu\n\n",
		opt->lambda, opt->gamma, opt->num_labels, opt->size_minibatch, opt->max_epochs,
		opt->eval_freq, opt->eval_max_acc_unimproved, random_seed);

	// Run algorithm
	matrix *W = NULL;
	if ((W = RDA(devID, opt)) == NULL) {
		printf("RDA failed!");
		return 2;
	}

	// Save to disk
	matrix_write(filename_W, W, false);

	// Clean-up
	free_rda_options(opt);
	matrix_free(W);

    return 0;
}
