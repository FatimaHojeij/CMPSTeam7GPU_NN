
NVCC = nvcc
NVCC_FLAGS = -O3 
BASE_OBJ = main.o verify.o matrix.o
OBJ_CPU = $(BASE_OBJ) kernel_cpu.o
OBJ_GPU0 = $(BASE_OBJ) kernel_gpu0.o
OBJ_GPU1 = $(BASE_OBJ) kernel_gpu1.o
OBJ_GPU2 = $(BASE_OBJ) kernel_gpu2.o
OBJ_GPU3 = $(BASE_OBJ) kernel_gpu3.o
OBJ_GPU4 = $(BASE_OBJ) kernel_gpu4.o
OBJ_GPU5 = $(BASE_OBJ) kernel_gpu5.o
OBJ_GPU6 = $(BASE_OBJ) kernel_gpu6.o
OBJ_GPU7 = $(BASE_OBJ) kernel_gpu7.o

EXE_CPU = spnn_cpu
EXE_GPU0 = spnn_gpu0
EXE_GPU1 = spnn_gpu1
EXE_GPU2 = spnn_gpu2
EXE_GPU3 = spnn_gpu3
EXE_GPU4 = spnn_gpu4
EXE_GPU5 = spnn_gpu5
EXE_GPU6 = spnn_gpu6
EXE_GPU7 = spnn_gpu7

default: $(EXE_CPU) $(EXE_GPU0) $(EXE_GPU1) $(EXE_GPU2) $(EXE_GPU3)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(EXE_CPU): $(OBJ_CPU)
	$(NVCC) $(NVCC_FLAGS) $(OBJ_CPU) -o $(EXE_CPU)

$(EXE_GPU0): $(OBJ_GPU0)
	$(NVCC) $(NVCC_FLAGS) $(OBJ_GPU0) -o $(EXE_GPU0)

$(EXE_GPU1): $(OBJ_GPU1)
	$(NVCC) $(NVCC_FLAGS) $(OBJ_GPU1) -o $(EXE_GPU1)

$(EXE_GPU2): $(OBJ_GPU2)
	$(NVCC) $(NVCC_FLAGS) $(OBJ_GPU2) -o $(EXE_GPU2)

$(EXE_GPU3): $(OBJ_GPU3)
	$(NVCC) $(NVCC_FLAGS) $(OBJ_GPU3) -o $(EXE_GPU3)

$(EXE_GPU4): $(OBJ_GPU4)
	$(NVCC) $(NVCC_FLAGS) $(OBJ_GPU4) -o $(EXE_GPU4)

$(EXE_GPU5): $(OBJ_GPU5)
	$(NVCC) $(NVCC_FLAGS) $(OBJ_GPU5) -o $(EXE_GPU5)

$(EXE_GPU6): $(OBJ_GPU6)
	$(NVCC) $(NVCC_FLAGS) $(OBJ_GPU6) -o $(EXE_GPU6)

$(EXE_GPU7): $(OBJ_GPU7)
	$(NVCC) $(NVCC_FLAGS) $(OBJ_GPU7) -o $(EXE_GPU7)

clean:
	rm -rf *.o spnn_*

