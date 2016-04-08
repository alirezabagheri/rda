/* Common IO Utilities
 * Alireza Bagheri Garakani (me@alirezabagheri.com), 12/22/2014
 * license: GNU GPLv3
 *
 * TODO:
 *  - better error checking on io operations (e.g, fread, fwrite, etc.); better clean-up on error
 */

#include "common_io.h"


/* Function reads matrix from disk.

     Input:
		filename	location of data filename
		transpose	boolean whether to transpose after read.

	 Output: matrix struct on success; otherwise, NULL.
*/
matrix *matrix_read(const char *filename, bool transpose){
	size_t i, j;
	bool is_BE;
	int rows, cols, *buff_i = NULL;
	float *buff_f = NULL;
	FILE *input_file = NULL;

	// Create matrix stuct
	matrix *mat = (matrix *)malloc(sizeof(matrix));

	// Check filename
	if (filename == NULL){
		printf("Failed to read; arguments must be non-NULL\n");
		matrix_free(mat);
		return NULL;
	}

	// Check if machine format is BE
	is_BE = is_bigendian();

	// Open file pointer
	if ((input_file = fopen(filename, "rb")) == NULL){
		printf("Can not open file %s!\n", filename);
		matrix_free(mat);
		return NULL;
	}

	// Read number of rows
	fread(&rows, sizeof(int), 1, input_file);
	if (!is_BE) rows = BSWAP32(rows); // Swap if not BE
	mat->row = (size_t) rows;

	// Read number of columns
	fread(&cols, sizeof(int), 1, input_file);
	if (!is_BE) cols = BSWAP32(cols); // Swap if not BE
	mat->col = (size_t) cols;

	printf("Reading matrix (%zu x %zu)%s: %s \t(machine is %s)\n", mat->row, mat->col,
		transpose ? " as transpose" : "", filename, is_BE ? "big-endian" : "little-endian");

	if (is_BE) {
		// Allocate  memory
		mat->data = (float *)malloc(sizeof(float) *  mat->row *  mat->col);
		if (transpose) buff_f = (float *)malloc(sizeof(float) *  mat->col);

		// Read each row of matrix
		for (i = 0; i < mat->row; i++) {
			if (transpose){
				fread(buff_f, sizeof(float), mat->col, input_file);
				for (j = 0; j < mat->col; j++) mat->data[(j * mat->row) + i] = buff_f[j];
			} else {
				fread(mat->data + (i * mat->col), sizeof(float), mat->col, input_file);
			}
			fseek(input_file, 4, SEEK_CUR); // expecting repeat of 'num columns' value; skip over.
		}

		if (transpose) free(buff_f);

	} else {
		// Read as int, flip endian, then cast to float.

		// Allocate  memory
		int *matrix_i = (int *)malloc(sizeof(int)* mat->row * mat->col);
		if (transpose) buff_i = (int *)malloc(sizeof(int) * mat->col);


		// Read each row of matrix
		for (i = 0; i < mat->row; i++) {
			if (transpose){
				fread(buff_i, sizeof(int), mat->col, input_file);
				for (j = 0; j < mat->col; j++) matrix_i[(j * mat->row) + i] = buff_i[j];
			} else {
				fread(matrix_i + (i * mat->col), sizeof(int), mat->col, input_file);
			}
			fseek(input_file, 4, SEEK_CUR); // expecting repeat of 'num columns' value; skip over.
		}

		if (transpose) free(buff_i);

		// Flip endian
		for (i = 0; i < mat->row * mat->col; i++) matrix_i[i] = BSWAP32(matrix_i[i]);

		// Cast to float;
		mat->data = (float *)matrix_i;
	}

	if (transpose){
		i = mat->col;
		mat->col = mat->row;
		mat->row = i;
	}

	// Close file pointer
	fclose(input_file);

	// Return
	return mat;
}


/* Function performs in-place permutation of indices on input vector (see "Fisher-Yates shuffle")

	Input:
		input	vector of indices
		length	length of vector
		seed	random seed

	Output: true on sucess
*/
bool helper_permutation(size_t *input, size_t length, unsigned int seed) {

	// Set generator seed
	srand(seed);

	// Set pointer to end of array
	size_t r, temp;
	for (size_t i = length - 1; i > 0; i--) {
		// Get random index in range [0,i] (TODO: find better rand)
		r = (size_t)((rand() / (double)RAND_MAX) * (length + 1));

		// Swap value at indices i and r
		temp = input[r]; input[r] = input[i]; input[i] = temp;
	}

	return true;
}


/* Function sets random seed. Only relevent when permute_rows for streaming matrix is enabled.

	Input:
		mat		streaming matrix struct
		seed	random seed

	Output: true on success.
*/
bool set_seed(matrix_s *mat, size_t seed) {
	mat->seed = seed;
	return true;
}


/* Function creates matrix stream s.t matrix can be read in by parts (set of rows)

	Input:
		filename			location of data filename
		permute_rows		indicate whether rows should be read randomly (without replacement) 
		loop_file			indicate whether should return to start of file upon reaching EOF.

	Output: streaming matrix struct
*/
matrix_s *matrix_stream_create(const char *filename, bool permute_rows, bool loop_file){
	int rows, cols;

	// Create matrix stuct
	matrix_s *mat = (matrix_s *)malloc(sizeof(matrix_s));
	mat->is_loop_file = loop_file;
	mat->is_permute_rows = permute_rows;
	mat->num_loops = 0;
	mat->seed = (size_t)(rand());
	mat->data = NULL;
	mat->row_idx = NULL;
	mat->perm = NULL;

	// Check if machine format is BE
	mat->is_BE = is_bigendian();

	// Check filename
	if (filename == NULL){
		printf("Failed to read; arguments must be non-NULL\n");
		matrix_free(mat);
		return NULL;
	}
	// Open file pointer
	if ((mat->fp = fopen(filename, "rb")) == NULL){
		printf("Can not open file %s!\n", filename);
		matrix_free(mat);
		return NULL;
	}

	// Read number of rows
	if (fread(&rows, sizeof(int), 1, mat->fp) != 1){
		printf("Error while reading or prematurely reached EOF; check file format.");
		matrix_free(mat);
		return NULL;
	}
	if (!mat->is_BE) rows = BSWAP32(rows); // Swap if not BE
	// Set values
	mat->row_total = (size_t)rows;
	mat->row_remaining = mat->row_total;
	mat->row_read = 0;
	mat->row = 0;

	// Read number of columns
	if (fread(&cols, sizeof(int), 1, mat->fp) != 1){
		printf("Error while reading or prematurely reached EOF; check file format.");
		matrix_free(mat);
		return NULL;
	}
	if (!mat->is_BE) cols = BSWAP32(cols); // Swap if not BE
	// Set values
	mat->col = (size_t)cols;

	// Additional handling if permute rows is enabled
	if (mat->is_permute_rows) {
		// Allocate and initialize index permutation array
		mat->perm = (size_t *)malloc(sizeof(size_t)* mat->row_total);
		for (size_t i = 0; i < mat->row_total; i++) mat->perm[i] = i;

		// To optimize fseek, we re-define the internal buffer size 
		// so that extra/too-litte data is NOT wastefully read to accomodate
		// a single row.
		fclose(mat->fp);
		if ((mat->fp = fopen(filename, "rb")) == NULL){
			printf("Can not open file %s!\n", filename);
			matrix_free(mat);
			return NULL;
		}
		setvbuf(mat->fp, NULL, _IOFBF, mat->col);
		fseek(mat->fp, 8, SEEK_CUR); // Seek to start of first row
	}

	printf("Opening %s%sstream for matrix (%zu x %zu): %s \t(machine is %s)\n", 
		permute_rows ? "row-permuting " : "", loop_file ? "file-looping " : "",
		mat->row_total, mat->col, filename, mat->is_BE ? "big-endian" : "little-endian");

	return mat;
}


/* Function creates matrix stream s.t matrix can be read in by parts (set of rows)

	Input:
		mat			streaming matrix struct
		rows		number of rows to read (positive integer)

	Output: true if desired rows could be read (or at least one row read if no file loop)

	NOTE:
		* no changes are made to struct if return value is false.
		* client must free by making call to matrix_free()
*/
bool matrix_stream_read(matrix_s *mat, size_t rows){
	size_t i;
	bool memsize_changed;

	// Handle error cases
	if (!matrix_isValid(mat) || rows == 0 || mat->row_remaining == 0) return false;


	if (mat->is_loop_file){
		memsize_changed = (mat->row != rows);
		mat->row = rows;
	} else {
		size_t rows_to_read = rows > mat->row_remaining ? mat->row_remaining : rows;
		memsize_changed = (mat->row != rows_to_read);
		mat->row = rows_to_read;
		mat->row_remaining -= mat->row;
	}

	// Allocate memory for index vector if requested rows has changed.
	if (memsize_changed){
		if (mat->row_idx != NULL) free(mat->row_idx);
		mat->row_idx = (size_t *)malloc(sizeof(size_t) *  mat->row);
	}


	if (mat->is_BE) {
		// Allocate memory for matrix data if requested rows has changed.
		if (memsize_changed){
			if (mat->data != NULL) free(mat->data);
			mat->data = (float *)malloc(sizeof(float)*  mat->row *  mat->col);
		}

		// Read each row of matrix
		for (i = 0; i < mat->row; i++) {
			mat->row_idx[i] = (mat->row_read + i) % mat->row_total;

			if (mat->is_permute_rows){
				if (mat->row_idx[i] == 0) {
					helper_permutation(mat->perm, mat->row_total, mat->seed + mat->num_loops);
					if (mat->row_read != 0) mat->num_loops++;
				}
				mat->row_idx[i] = mat->perm[mat->row_idx[i]];
				fseek(mat->fp, 4 * (2 + mat->row_idx[i] * (mat->col + 1)), SEEK_SET);

			} else if (mat->is_loop_file){
				// If EOF is reached, seek to start of first row.
				if (mat->row_idx[i] == 0 && mat->row_read != 0) {
					fseek(mat->fp, 8, SEEK_SET);
					mat->num_loops++;
				}
			}

			if (fread(mat->data + (i * mat->col), sizeof(float), mat->col, mat->fp) != mat->col){
				printf("Error while reading or prematurely reached EOF; check file format.");
				matrix_free(mat);
				return false;
			}

			// expecting repeat of 'num columns' value; skip over. If permute_rows is enabled,
			// don't worry about this seek b/c we will re-seek from start of file anyway.
			if (!mat->is_permute_rows) fseek(mat->fp, 4, SEEK_CUR); 
		}

	} else {
		// Read as int, flip endian, then cast to float.

		// Allocate memory for matrix data if requested rows has changed.
		int *matrix_i = NULL;
		if (memsize_changed){
			if (mat->data != NULL) free(mat->data);
			matrix_i = (int *)malloc(sizeof(int)* mat->row * mat->col);
		} else {
			matrix_i = (int *)(mat->data);
		}

		// Read each row of matrix
		for (i = 0; i < mat->row; i++) {
			mat->row_idx[i] = (mat->row_read + i) % mat->row_total;

			if (mat->is_permute_rows){
				if (mat->row_idx[i] == 0){
					helper_permutation(mat->perm, mat->row_total, mat->seed + mat->num_loops);
					if (mat->row_read != 0) mat->num_loops++;
				}
				mat->row_idx[i] = mat->perm[mat->row_idx[i]];
				fseek(mat->fp, 4 * (2 + mat->row_idx[i] * (mat->col + 1)), SEEK_SET);

			}
			else if (mat->is_loop_file){
				// If EOF is reached, seek to start of first row.
				if (mat->row_idx[i] == 0 && mat->row_read != 0) {
					fseek(mat->fp, 8, SEEK_SET);
					mat->num_loops++;
				}
			}

			if (fread(matrix_i + (i * mat->col), sizeof(int), mat->col, mat->fp) != mat->col){
				printf("Error while reading or prematurely reached EOF; check file format.");
				matrix_free(mat);
				return false;
			}

			// expecting repeat of 'num columns' value; skip over. If permute_rows is enabled,
			// don't worry about this seek b/c we will re-seek from start of file anyway.
			if (!mat->is_permute_rows) fseek(mat->fp, 4, SEEK_CUR);
		}

		// Flip endian
		for (i = 0; i < mat->row * mat->col; i++) matrix_i[i] = BSWAP32(matrix_i[i]);

		// Cast to float;
		mat->data = (float *)matrix_i;
	}


	// Check if done reading file
	if (!mat->is_loop_file && mat->row_remaining == 0){
		fclose(mat->fp); 
		mat->fp = NULL;
	}

	mat->row_read += mat->row;
	
	// Return
	return true;
}


/* Function writes matrix to disk.

     Input:
		filename        location of data filename to write to
		matrix	        matrix to write
		transpose	boolean whether to write matrix after transpose
	 
	 Output: true on success
 */
bool matrix_write(const char *filename, const matrix *mat, bool transpose){
	int rows, cols, val, *buff_i = NULL;
	size_t i, j;
	bool is_BE;
	float *buff_f = NULL;
	FILE *output_file = NULL;

	if (filename == NULL || !matrix_isValid(mat)){
		printf("Failed to write; arguments must be non-NULL\n");
		return false;
	}

	// Check if machine format is BE
	is_BE = is_bigendian();

	// Open file pointer	
	if ((output_file = fopen(filename, "wb")) == NULL){
		printf("Can not open file %s!\n", filename);
		return false;
	}

	printf("Writing matrix (%zu x %zu)%s: %s \t(machine is %s)\n", mat->row, mat->col,
		transpose ? " after transpose" : "", filename, is_BE ? "big-endian" : "little-endian");

	if (is_BE){

		if (transpose) {
			// Allocate memory
			buff_f = (float *)malloc(sizeof(float) * mat->row);

			fwrite(&(mat->col), sizeof(int), 1, output_file);
			// Write each row to file
			for (j = 0; j < mat->col; j++){
				fwrite(&(mat->row), sizeof(int), 1, output_file);
				// Write this row
				for (i = 0; i < mat->row; i++) buff_f[i] = mat->data[(i * mat->col) + j];
				fwrite(buff_f, sizeof(float), mat->row, output_file);
			}

			free(buff_f);
		} else {
			fwrite(&(mat->row), sizeof(int), 1, output_file);
			// Write each row to file
			for (i = 0; i < mat->row; i++){
				fwrite(&(mat->col), sizeof(int), 1, output_file);
				// Write this row
				fwrite(mat->data + (i * mat->col), sizeof(float), mat->col, output_file);
			}
		}
		
	} else {
		// Conversion from 'size_t -> int' can be problematic 
		rows = BSWAP32((int)(mat->row));
		cols = BSWAP32((int)(mat->col));

		if (transpose) {
			// Allocate memory
			buff_i = (int *)malloc(sizeof(int) * mat->row);
			
			fwrite(&cols, sizeof(int), 1, output_file);
			// Write each row to file
			for (j = 0; j < mat->col; j++){
				fwrite(&rows, sizeof(int), 1, output_file);
				// Write this row
				for (i = 0; i < mat->row; i++) {
					memcpy(&val, mat->data + (i * mat->col) + j, sizeof(float));
					val = BSWAP32(val);
					buff_i[i] = val;
				}
				fwrite(buff_i, sizeof(int), mat->row, output_file);
			}

			free(buff_i);
		} else {
			// Allocate memory
			buff_i = (int *)malloc(sizeof(int) * mat->col);

			fwrite(&rows, sizeof(int), 1, output_file);
			// Write each row to file
			for (i = 0; i < mat->row; i++){
				fwrite(&cols, sizeof(int), 1, output_file);
				// Write this row
				for (j = 0; j < mat->col; j++){
					memcpy(&val, mat->data + (i * mat->col) + j, sizeof(float));
					val = BSWAP32(val);
					buff_i[j] = val;
				}
				fwrite(buff_i, sizeof(int), mat->col, output_file);
			}

			free(buff_i);
		}
	}

	// Close file pointer
	fclose(output_file);

	return true;
}


/* Function frees matrix or streaming matrix struct
	Input:
		mat        matrix (or streaming matrix) struct

	Output: true on success
*/
bool matrix_free(matrix *mat){
	if (mat != NULL){
		if (mat->data != NULL) free(mat->data);
		free(mat);
	}
	return true;
}
bool matrix_free(matrix_s *mat) {
	if (mat != NULL) {
		if (mat->data != NULL) free(mat->data);
		if (mat->fp != NULL) fclose(mat->fp);
		free(mat);
	}
	return true;
}

/* Function checks if matrix or streaming matrix pointer is valid.
	Input:
		mat        matrix (or streaming matrix) struct

	Output: true if valid
*/
bool matrix_isValid(const matrix_s *mat) {
	return mat != NULL && mat->fp != NULL && ((mat->data != NULL && mat->row_idx != NULL) || mat->row == 0);
}
bool matrix_isValid(const matrix *mat) {
	return mat != NULL && mat->data != NULL;
}