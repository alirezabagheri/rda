/* Header for Common IO Utilities
 * Alireza Bagheri Garakani (me@alirezabagheri.com), 12/22/2014
 * license: GNU GPLv3
 */

#ifndef COMMON_IO_H
#define COMMON_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>

// Specify the size in bytes of each element in binary data files.
// Expecting 'sizeof(int) == FILE_BIN_SIZE'
#define FILE_BIN_SIZE	4

// Used to determine endianness of machine at runtime (data is also BE format)
const unsigned int _int_one = 1;
#define is_bigendian() ( (*(char*)&_int_one) == 0 )
#define BSWAP32(x) (((x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | ((x >> 24) & 0x000000ff)))

// Matrix struct
typedef struct {
	size_t row, col;
	float *data;
} matrix;

// Streaming Matrix struct
typedef struct {
	size_t row, col, row_total, row_remaining, row_read, num_loops, seed, *row_idx, *perm;
	float *data;
	// keeping state
	FILE *fp;
	bool is_BE, is_permute_rows, is_loop_file;
} matrix_s;


matrix *matrix_read(const char *filename, bool transpose);
bool matrix_free(matrix *mat);
bool matrix_isValid(const matrix *mat);

matrix_s *matrix_stream_create(const char *filename, bool permute_rows, bool loop_file);
bool matrix_stream_read(matrix_s *mat, size_t rows);
bool set_seed(matrix_s *mat, size_t seed);
bool matrix_free(matrix_s *mat);
bool matrix_isValid(const matrix_s *mat);

bool matrix_write(const char *filename, const matrix *mat, bool transpose);


#endif