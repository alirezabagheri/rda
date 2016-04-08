function matrix = bin2mat(input_filename)
% BIN2MAT Convert matrix from binary format to matlab format
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

% Check input file
if ~exist(fullfile(cd, input_filename), 'file')
    error('input file does not exist.');
end

% Open file pointer
fid = fopen(input_filename, 'r', 'ieee-be');

% Read "number of frames" (this translates to matrix rows)
mat_rows = fread(fid, 1, 'int32', 0, 'ieee-be');

% Read "number of dimensions" (this translates to matrix cols)
mat_cols = fread(fid, 1, 'int32', 0, 'ieee-be');

matrix = zeros(mat_rows,mat_cols);

% Read matrix
for i = 1:mat_rows-1
    % read num_cols entries and insert into matrix row
    matrix(i,:) = fread(fid, mat_cols, 'single', 0, 'ieee-be');
    % use the repeat num_cols value to check against what we know; then discard.
    if (mat_cols ~= fread(fid, 1, 'int32', 0, 'ieee-be'))
      error('matrix rows are not the same length');
    end
end
% read num_cols entries and insert into matrix row
matrix(mat_rows,:) = fread(fid, mat_cols, 'single', 0, 'ieee-be');

end
