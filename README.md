#####Mulitnomial Logistic Regression using Regularized Dual Averaging (RDA)
* Stochastic, Group-Lasso Regularized, GPU-accelerated

#####Platform: 
* Linux 64-bit (tested on Centos 6.6)

#####Dependencies:
* CUDA Toolkit (tested on version 7.0)

#####Getting Started: 
1. Ensure dependencies are installed on your system. 

* Compile source code: 
 1. Navigate to the 'common' sub-directory of the package and execute **make** to generate 'common/lib/libcommon.so'. This library contains routines to read/write binary matrix/vector data.

 * Navigate to the 'rda' sub-directory of the package and execute **make** to generate binary executable 'rda/RDA'. 

2. Set environment variable 'LD\_LIBRARY\_PATH' to include CUDA lib directory and libcommon.so parent directory. 

3. Execute RDA executable without arguments to see usage (shown below for reference):
      > Usage: ./RDA <device-id> <path:W> <path:X> <path:y> <path:X_h> <path:y_h> <float:lambda> <float:gamma> <size:minibatch_size> <size:max_epochs> <float:eval_freq> <size:eval_max_acc_unimproved> [<size: random_seed>]

      >    device-id - gpu device id (e.g., 0 or 1)
      
      >    W - output matrix: D x K
      
      >    X - data matrix N x D
      
      >    y - label vector (N x 1), where each element is in [0 ... K-1]
      
      >    X\_h - heldout data matrix: M x D
      
      >    y\_h - heldout label vector (M x 1), where each element is in [0 ... K-1]
      
      >    lambda - group-penalty coeff
      
      >    gamma - dual averaging parameter
      
      >    num\_labels - number of unique labels K
      
      >    minibatch\_size - size of minibatch used for each gradient step
      
      >    max\_epochs - maximum number of epochs allowed for training
      
      >    eval\_freq - frequency of heldout evaluation (e.g., use 1.0 for eval after each training epoch)
      
      >    eval\_max\_acc_unimproved - treshold number of evaluations before terminating due to convergence (e.g., 10)
      
      >    random_seed - set seed effecting permutation of training data. (optional, default: 1)


4. Prepare input data (separate binary files for features and labels for both train and heldout datasets). Refer to 'matlab' sub-directory for conversion to/from Matrix matrix. Files in sub-directory also contain description of binary file format s.t. an alternative interface (e.g. python, C, etc.) can be implemented instead.

5. Run **RDA** executable. Sample output shown below using MNIST-8M dataset (train contains 6.75M samples generated from transformations of original MNIST 50K training samples (randomly selected out of 60K); the remaining 10K training data not selected is used as heldout data with no transformation).

      > ./RDA 0 output.bin /dev/shm/data/X.bin /dev/shm/data/y.bin /dev/shm/data/X_heldout.bin /dev/shm/data/y_heldout.bin .1 1 100 100 10 1 10 7

      > Opening row-permuting file-looping stream for matrix (6750000 x 785): /dev/shm/data/X.bin       (machine is little-endian)

      > Reading matrix (1 x 6750000): /dev/shm/data/y.bin       (machine is little-endian)

      > Opening file-looping stream for matrix (10000 x 785): /dev/shm/data/X_heldout.bin       (machine is little-endian)

      > Reading matrix (1 x 10000): /dev/shm/data/y_heldout.bin         (machine is little-endian)

      > lambda = 0.1, gamma = 1, num_labels = 100, size_minibatch = 100, max_epochs = 10, eval_freq = 1, eval_max_acc_unimproved = 10, random_seed = 7


      > Running algorithm.

      > \*\*\*\*\*\*\*\*\*\*

      > iteration 67500 (epoch 1),  non-zero covariates: 709 (90.20%),  heldout accuracy: 87.69% (best 87.69%; unimproved 0)

      > [...]

      > 


#####Notes:
* if reading random rows of train data matrix causes excessive IO, consider either [1] moving input file to /dev/shm (assuming data fits in main memory), or [2] disabling row-permutation option (see call to matrix\_stream\_create() function in main). 

#####References: 

* L. Xiao. Dual averaging methods for regularized stochastic learning and online optimization. Technical Report MSR-TR-2010-23, Microsoft Research, 2010.
* Yuan M, Lin Y. Model selection and estimation in regression with grouped variables. Journal of the Royal Statistical Society, Series B. 2007;68(1):49–67.
* H. Yang, Z. Xu, I. King, and M. Lyu, "Online learning for group lasso," in International Conference on Machine Learning (ICML'10), 2010. 