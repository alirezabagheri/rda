CC = /usr/local/cuda/bin/nvcc
CFLAGS = --compiler-options '-O3 -Wall' -Iinclude/ -I../common/include --relocatable-device-code=false -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -link
LDFLAGS = -L../common/lib -L/usr/local/cuda-6.5/lib64 -lcommon -lcublas
DEPS = 
OBJ = src/RDA.o
TARGET  = RDA

all: RDA

%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

# Tool invocations
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

# Other Targets
clean:
	rm -f $(OBJ) $(TARGET)
