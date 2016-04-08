CC = /usr/local/cuda/bin/nvcc
CFLAGS = --compiler-options '-O3 -fPIC -Wall' -Iinclude/
LDFLAGS = -shared
CFLAGS_TEST = -g -Llib/ -lcommon -Iinclude/
DEPS = include/*
OBJ = src/common_io.o
TARGET  = lib/libcommon.so

%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(OBJ) $(TARGET)

test_io: test/test_io.o
	$(CC) -o test/$@ $^ $(CFLAGS_TEST)
	export LD_LIBRARY_PATH=$(shell pwd)/lib; cd test && ./$@; cd ..
