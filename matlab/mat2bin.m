function mat2bin(output_filename, matrix)
% MAT2BIN Convert matrix from matlab format to binary format
% Alireza Bagheri Garakani (me@alirezabagheri.com), 2/20/2016
% license: GNU GPLv3
%
% Description of binary format:
%   Each file contains a single matrix (or vector) written with big-endian 
%   byte ordering. The binary file is composed of 4-byte entries. The first
%   entry represents the number of matrix rows (N). The rest of the file 
%   contains, for each row N, an entry representing number of columns (M)
%   followed by M entries represented by IEEE single-precision floats.
%
%   E.G., the following matlab matrix would be written as follows: 
%    "[1 2; 3 4]" ---> 
%    "2(int32) 2(int32) 1.0(single) 2.0(single) 2(int32) 3.0(single) 4.0(single) 

% Check input matrix
if ~isa(matrix,'single') || issparse(matrix) || ~ismatrix(matrix)
    error('matrix should be dense, 2D, single precision floating point values');
end

% Check target output file
if exist(fullfile(cd, output_filename), 'file')
    error('output file exists; delete and re-run');
end

mat_rows = size(matrix,1);
mat_cols = size(matrix,2);

% Open file pointer
fid = fopen(output_filename, 'w');

% Write "number of frames" (this translates to matrix rows)
fwrite(fid, mat_rows, 'int32', 0, 'ieee-be');

% Write matrix
for i = 1:mat_rows
    % write number of feature dimensions (i.e. matrix columns)
    fwrite(fid, mat_cols, 'int32', 0, 'ieee-be');
    % write data
    fwrite(fid, matrix(i,:), 'single', 0, 'ieee-be');
end

end
