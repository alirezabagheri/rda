/* Unit tests for Common IO Utilities
 * Alireza Bagheri Garakani (me@alirezabagheri.com), 12/22/2014
 * license: GNU GPLv3
 */

#include <stdio.h>
#include <stdlib.h>
#include "common_io.h"

const char * filename = "data.binary";
const char * filename2 = "data2.binary";

void print_matrix(const matrix *mat, int rows, int cols){
	int i, j;
	for (i = 0; i < rows; i++) {
                for (j = 0; j < cols; j++) printf("%f,", mat->data[i*cols + j]);
                printf("\n");
        }
}

int invalid_transpose(const matrix *mat, const matrix *mat2, int rows, int cols){
	int i, j;
	for (i = 0; i < rows; i++) for (j = 0; j < cols; j++){
		if (mat->data[(i * cols) + j] != mat2->data[(j * rows) + i]){
			return 1;
		}
    }
	return 0;
}

int not_equal(const matrix *mat, const matrix *mat2, int rows, int cols){
	int i;
	for (i = 0; i < rows * cols; i++){
		if (mat->data[i] != mat2->data[i]) return 1;
	}
	return 0;
}


// Read transpose
void test1(){
	printf("TEST-1\n");
	matrix *mat, *mat2;

	mat = matrix_read(filename, false);
	mat2 = matrix_read(filename, true);

	if (mat->row != mat2->col || mat->col != mat2->row) {
		printf("FAIL\n");
		return;
	}

	if (invalid_transpose(mat,mat2,mat->row,mat->col)){
		printf("FAIL\n");
		return;
	}

	matrix_free(mat);
	matrix_free(mat2);

	printf("SUCCESS\n");

}

// Read + write (no transpose)
void test2(){
	printf("TEST-2\n");
	matrix *mat, *mat2;

	mat = matrix_read(filename, false);
	if (!matrix_write(filename2, mat, false)){
		printf("FAIL\n");
		return;
	}
	mat2 = matrix_read(filename2, false);

	if (not_equal(mat, mat2, mat->row, mat->col)){
		printf("FAIL\n");
        return;
    }

	matrix_free(mat);
	matrix_free(mat2);

	printf("SUCCESS\n");
}

//read + write (trans) + read
void test3(){
	printf("TEST-3\n");
	matrix *mat, *mat2;

    mat = matrix_read(filename, false);
    matrix_write(filename2, mat, true);
    mat2 = matrix_read(filename2, false);

    if (invalid_transpose(mat, mat2, mat->row, mat->col)){
        printf("FAIL\n");
        return;
    }

	matrix_free(mat);
	matrix_free(mat2);

    printf("SUCCESS\n");
}

void timing(){
	printf("TIMING\n");
	matrix *mat, *mat2;

	system("date");
	mat = matrix_read(filename, false);

	system("date");
	mat2 = matrix_read(filename, true);

	system("date");
	matrix_write(filename2, mat, false);

	system("date");
	matrix_write(filename2, mat, true);
	
	system("date");

	matrix_free(mat);
	matrix_free(mat2);
}

void test10(){
	printf("TEST-10\n");
	matrix_s *mat = matrix_stream_create(filename, true, true);

	matrix_stream_read(mat, 1);

	matrix_stream_read(mat, 2);

	matrix_stream_read(mat, 2);

	matrix_stream_read(mat, 10);

}

int main(){
	// Tests for matrix_read / matrix_write
	printf("===== Tests for matrix_read / matrix_write\n");
	timing();
	test1();
	test2();
	test3();

	// Tests for streaming read/write. (for step-debugging)
	// test10();

	return 0;
}
