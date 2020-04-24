
// row+1; swapping; nnzidx; syncthreads
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"

#include<string.h>
#define THRESHOLD 0.000001
#define YMAX 32
#define threads 32
#define BLOCK_DIM 1024
#define CAPACITY 25498020

//__constant__ unsigned int u_Max;


__global__ void spmspm(COOMatrix *result, CSRMatrix* A, CSCMatrix B, float bias) {
	unsigned int r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int c = blockIdx.x*blockDim.x + threadIdx.x;


	if (r < A->numRows && c < B.numCols) {
		unsigned int rowPtrA = A->rowPtrs[r];
		unsigned int nnzA = A->rowPtrs[r + 1] - rowPtrA;

		unsigned int colPtrB = B.colPtrs[c];
		unsigned int nnzB = B.colPtrs[c + 1] - colPtrB;
		if (nnzA > 0 && nnzB > 0) { // if a row is not all zeros , we do computation otherwise we skip row

			float sum = 0.0f;
			unsigned int ia = 0, ib = 0;
			while (ia < nnzA && ib < nnzB) { // loops over all non zeros from A and B and stop when there is no more non zero

				unsigned int colIdx = A->colIdxs[rowPtrA + ia]; //single item col index from A
				unsigned int rowIdx = B.rowIdxs[colPtrB + ib]; //single item row index from B
				if (rowIdx < B.nnz && colIdx < A->nnz) {
					if (colIdx < rowIdx) {
						ia++;
					}
					else if (colIdx > rowIdx) {
						ib++;
					}
					else {
						sum += A->values[rowPtrA + ia] * B.values[ib + colPtrB];// do the multiplication of the row that matches the column
						ia++;
						ib++;
					}
				}

			}
			if (sum > THRESHOLD || sum < -THRESHOLD) { //if not smaller than abs(threshold)
				sum += bias; //add to it the bias
				//Remove negative and zero values
				if (sum > 0) {//if end result is positive otherwise I also do not want to add it to result
					if (sum > YMAX) { //make sure it is on an upper limit
						sum = YMAX;
					}
					unsigned int nnzIndxTemp = atomicAdd(&(result->nn), 1); //counts how many non zero elements I have
					result->rowIdxs[nnzIndxTemp] = r;
					result->colIdxs[nnzIndxTemp] = c;
					result->values[nnzIndxTemp] = sum;
				}
			}


		}
	}

}


__global__ void histogram_private_kernel(COOMatrix* coo_d, unsigned int* rowPtrs) {


	unsigned int t = blockDim.x*blockIdx.x + threadIdx.x;

	if (t < coo_d->nnz) {
		unsigned int rIdx = coo_d->rowIdxs[t];
		atomicAdd(&(rowPtrs[rIdx]), 1);
	}



}


__global__ void scan_kernel(unsigned int* input, CSRMatrix* output, unsigned int* partialSums, unsigned int N) {

	// TODO


	unsigned int segment = 2 * blockDim.x * blockIdx.x;
	unsigned int i = segment + threadIdx.x;

	__shared__ unsigned int input_s[2 * BLOCK_DIM];

	int tid = threadIdx.x;


	if (i < N)
	{
		input_s[tid] = input[i];
	}
	else
	{
		input_s[tid] = 0;
	}
	if (i + BLOCK_DIM < N)
	{
		input_s[tid + BLOCK_DIM] = input[i + BLOCK_DIM];
	}
	else
	{
		input_s[tid + BLOCK_DIM] = 0;
	}
	__syncthreads();


	//reduction step
	for (unsigned int stride = 1; stride <= BLOCK_DIM; stride *= 2)
	{
		int index = (threadIdx.x + 1) * 2 * stride - 1;
		if (index < 2 * BLOCK_DIM)
			input_s[index] += input_s[index - stride];
		__syncthreads();
	}

	//save partial sum
	if (threadIdx.x == 0)
	{
		partialSums[blockIdx.x] = input_s[2 * BLOCK_DIM - 1];
		input_s[2 * BLOCK_DIM - 1] = 0.0f;

	}

	__syncthreads();

	//post reduction step
	for (unsigned int stride = BLOCK_DIM; stride > 0; stride /= 2)
	{
		int index = (threadIdx.x + 1) * 2 * stride - 1;

		if (index < 2 * BLOCK_DIM)
		{
			//add then swap
			unsigned int temp = input_s[index];
			input_s[index] += input_s[index - stride];
			input_s[index - stride] = temp;
		}

		__syncthreads();
	}


	if (i < N)
	{
		output->rowPtrs[i] = input_s[tid];
	}
	if (i + BLOCK_DIM < N)
	{
		output->rowPtrs[i + BLOCK_DIM] = input_s[tid + BLOCK_DIM];
	}

}

__global__ void add_kernel(CSRMatrix* output, unsigned int* partialSums, unsigned int N) {

	// TODO
	unsigned int i = 2 * blockIdx.x*blockDim.x + threadIdx.x;
	if (blockIdx.x != 0) {
		if (i < N) {
			output->rowPtrs[i] += partialSums[blockIdx.x];
		}
		if (i + BLOCK_DIM < N) {
			output->rowPtrs[i + BLOCK_DIM] += partialSums[blockIdx.x];
		}
	}

}
//output_d rowptrs n = numrows +1
void scan_gpu_d(unsigned int* input_d, unsigned int* output_d, unsigned int N) {

	// Configurations
	const unsigned int numThreadsPerBlock = BLOCK_DIM;
	const unsigned int numElementsPerBlock = 2 * numThreadsPerBlock;
	const unsigned int numBlocks = (N + numElementsPerBlock - 1) / numElementsPerBlock;

	// Allocate partial sums

	unsigned int *partialSums_d;
	cudaMalloc((void**)&partialSums_d, numBlocks * sizeof(unsigned int));
	cudaDeviceSynchronize();


	scan_kernel << < numBlocks, numThreadsPerBlock >> > (input_d, output_d, partialSums_d, N);
	cudaDeviceSynchronize();


	// Scan partial sums then add
	if (numBlocks > 1) {

		// Scan partial sums
		scan_gpu_d(partialSums_d, partialSums_d, numBlocks);

		// Add scanned sums
		add_kernel << < numBlocks, numThreadsPerBlock >> > (output_d, partialSums_d, N);

	}

	// Free memory
	cudaFree(partialSums_d);
	cudaDeviceSynchronize();

}


__global__ void Binning_kernel(COOMatrix* coo_d, CSRMatrix* csr_d, unsigned int* rowPtrsBin) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;

	if (i < coo_d->nnz) {
		unsigned int row = coo_d->rowIdxs[i];
		unsigned int col = coo_d->colIdxs[i];
		float val = coo_d->values[i];
		unsigned int init = csr_d->rowPtrs[row];
		unsigned int nnzIdx = atomicAdd(&rowPtrsBin[row], 1);
		csr_d->colIdxs[nnzIdx + init] = col;
		csr_d->values[nnzIdx + init] = val;
	}

}

__global__ void  sorting_kernel(CSRMatrix* csr_d) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	if (i < csr_d->numRows) {
		unsigned int rowPtrA = csr_d->rowPtrs[i];
		unsigned int nnzA = csr_d->rowPtrs[i + 1] - csr_d->rowPtrs[i];

		if (nnzA > 0) {
			for (unsigned int j = 0; j < nnzA - 1;++j) {

				for (unsigned int k = 0; k < nnzA - j - 1; ++k) {

					unsigned int l_0 = k + csr_d->rowPtrs[i];
					unsigned int l_1 = l_0 + 1;

					if (csr_d->colIdxs[l_0] > csr_d->colIdxs[l_1]) {
						//swap col
						unsigned int tmp = csr_d->colIdxs[l_0];
						csr_d->colIdxs[l_0] = csr_d->colIdxs[l_1];
						csr_d->colIdxs[l_1] = tmp;
						//swap float
						float valtmp = csr_d->values[l_0];
						csr_d->values[l_0] = csr_d->values[l_1];
						csr_d->values[l_1] = valtmp;
					}
				}
			}
		}

	}

}

//converts from CSRMatrix to Vector and a vector of indices where the row is not all zeros
void findNonzeroRows(Vector* v, CSRMatrix* A) {
	unsigned int nnz = 0;
	for (unsigned int r = 0; r < A->numRows; ++r) {
		unsigned int rowPtrA = A->rowPtrs[r];
		unsigned int nnzA = A->rowPtrs[r + 1] - rowPtrA;
		if (nnzA > 0) {
			if (nnz >= v->capacity) {
				expandVectorCapacity(v, 2 * v->capacity);
			}
			v->data[nnz] = r;
			++nnz;
		}
	}
	v->nnz = nnz;
}

// //to be removed
// COOMatrix* createEmptyCOO(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
// 	COOMatrix *coo = (COOMatrix *)malloc(sizeof(COOMatrix));
// 	coo->rowIdxs = (unsigned int *)malloc(capacity * sizeof(unsigned int));
// 	coo->colIdxs = (unsigned int *)malloc(capacity * sizeof(unsigned int));
// 	coo->values = (float *)malloc(capacity * sizeof(float));
// 	coo->numRows = numRows;
// 	coo->numCols = numCols;
// 	coo->nnz = 0;
// 	coo->capacity = CAPACITY;
// 	for (unsigned int i = 0; i < coo->capacity;++i) {
// 		coo->rowIdxs[i] = 0;
// 		coo->colIdxs[i] = 0;
// 		coo->values[i] = 0.0f;
// 	}
// 	return coo;
// }


void convertCOOtoCSR(COOMatrix* coo_d, CSRMatrix* csr_d){

	unsigned int  *rowPtrstmp_d;
	unsigned int *rowPtrstmp;
	rowPtrstmp = (unsigned int *)malloc((cootmp.numRows + 1) * sizeof(unsigned int));
	cudaMalloc((void**)&rowPtrstmp_d, (cootmp.numRows + 1) * sizeof(unsigned int));
	cudaMemcpy(rowPtrstmp_d, rowPtrstmp, (cootmp.numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);

	COOMatrix cootmp;
	cudaMemcpy(&cootmp, coo_d, sizeof(COOMatrix), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
	
	
	cudaMemset(&csr_d->nnz, cootmp.nnz, sizeof(unsigned int));

	for(int i = 0 ; i < cootmp.numRows+1; ++i){
		cudaMemset((void **)&(rowPtrstmp_d[i]),0,sizeof(unsigned int));
	}
	//calling histogram to fill rowPtrs of inBuffer
	unsigned int numThreadsPerBlock = 1024;
	unsigned int numBlocks = (cootmp.nnz+ numThreadsPerBlock - 1) / numThreadsPerBlock;

	//initializing rowstmp and rowstmp_d
	histogram_private_kernel << < numBlocks, numThreadsPerBlock >> > (coo_d, rowPtrstmp_d);

	cudaDeviceSynchronize();

	//calling the scan kernel to scan kernel ptrs
	const unsigned int numElementsPerBlock = 2 * numThreadsPerBlock;
	numBlocks = ((cootmp.numRows + 1) + numElementsPerBlock - 1) / numElementsPerBlock;

	// Allocate partial sums
	unsigned int *partialSums_d;
	cudaMalloc((void**)&partialSums_d, numBlocks * sizeof(unsigned int));
	cudaDeviceSynchronize();

	// Call kernel
	scan_kernel << < numBlocks, numThreadsPerBlock >> > (rowPtrstmp_d, csr_d, partialSums_d, cootmp.numRows + 1);

	cudaDeviceSynchronize();
	// Scan partial sums then add

	if (numBlocks > 1) {

		// Scan partial sums
		scan_gpu_d(partialSums_d, partialSums_d, numBlocks);

		// Add scanned sums
		add_kernel << < numBlocks, numThreadsPerBlock >> > (csr_d, partialSums_d, cootmp.numRows + 1);

	}

	cudaDeviceSynchronize();


	// Free memory

	cudaFree(partialSums_d);


	//Binning
	for(int i = 0 ; i < cootmp.numRows+1; ++i){
		cudaMemset((void **)&(rowPtrstmp_d[i]),0,sizeof(unsigned int));
	}

	cudaDeviceSynchronize();

	numBlocks = (cootmp.nnz + numThreadsPerBlock - 1) / numThreadsPerBlock;
	Binning_kernel << < numBlocks, numThreadsPerBlock >> > (coo_d, csr_d, rowPtrstmp_d);

	cudaDeviceSynchronize();


	//Sorting
	numBlocks = ((cootmp.numRows + 1) + numThreadsPerBlock - 1) / numThreadsPerBlock;
	sorting_kernel << < numBlocks, numThreadsPerBlock >> > (csr_d);

	cudaDeviceSynchronize();

	cudaFree(rowPtrstmp_d);
	free(rowPtrstmp);	




}

COOMatrix* createEmptyCOO_d(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    COOMatrix cooShadow;
    cooShadow.numRows = numRows;
    cooShadow.numCols = numCols;
    cooShadow.nnz = 0;
    cooShadow.capacity = capacity;
    cudaMalloc((void**) &cooShadow.rowIdxs, capacity*sizeof(unsigned int));
    cudaMalloc((void**) &cooShadow.colIdxs, capacity*sizeof(unsigned int));
    cudaMalloc((void**) &cooShadow.values, capacity*sizeof(float));
    COOMatrix* coo_d;
    cudaMalloc((void**) &coo_d, sizeof(COOMatrix));
    cudaMemcpy(coo_d, &cooShadow, sizeof(COOMatrix), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return coo_d;
}

void copyCOOfromGPU(COOMatrix* coo_d, COOMatrix* coo) {
    COOMatrix cooShadow;
    cudaMemcpy(&cooShadow, coo_d, sizeof(COOMatrix), cudaMemcpyDeviceToHost);
    assert(coo->numRows == cooShadow.numRows);
    assert(coo->numCols == cooShadow.numCols);
    assert(coo->capacity >= cooShadow.nnz);
    coo->nnz = cooShadow.nnz;
    cudaMemcpy(coo->rowIdxs, cooShadow.rowIdxs, cooShadow.nnz*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(coo->colIdxs, cooShadow.colIdxs, cooShadow.nnz*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(coo->values, cooShadow.values, cooShadow.nnz*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

CSRMatrix* createEmptyCSR_d(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    CSRMatrix csrShadow;
    csrShadow.numRows = numRows;
    csrShadow.numCols = numCols;
    csrShadow.nnz = 0;
    csrShadow.capacity = capacity;
    cudaMalloc((void**) &csrShadow.rowPtrs, (numRows + 1)*sizeof(unsigned int));
    cudaMalloc((void**) &csrShadow.colIdxs, capacity*sizeof(unsigned int));
    cudaMalloc((void**) &csrShadow.values, capacity*sizeof(float));
    CSRMatrix* csr_d;
    cudaMalloc((void**) &csr_d, sizeof(CSRMatrix));
    cudaMemcpy(csr_d, &csrShadow, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return csr_d;
}

void copyCSRtoGPU(CSRMatrix* csr, CSRMatrix* csr_d) {
    CSRMatrix csrShadow;
    cudaMemcpy(&csrShadow, csr_d, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    assert(csrShadow.numRows == csr->numRows);
    assert(csrShadow.numCols == csr->numCols);
    assert(csrShadow.capacity >= csr->nnz);
    csrShadow.nnz = csr->nnz;
    cudaMemcpy(csrShadow.rowPtrs, csr->rowPtrs, (csr->numRows + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrShadow.colIdxs, csr->colIdxs, csr->nnz*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrShadow.values, csr->values, csr->nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

CSCMatrix* createCSCfromCSC_d(CSCMatrix* csc) {
    CSCMatrix cscShadow;
    cscShadow.numRows = csc->numRows;
    cscShadow.numCols = csc->numCols;
    cscShadow.nnz = csc->nnz;
    cscShadow.capacity = csc->capacity;
    cudaMalloc((void**) &cscShadow.colPtrs, (csc->numCols + 1)*sizeof(unsigned int));
    cudaMalloc((void**) &cscShadow.rowIdxs, csc->capacity*sizeof(unsigned int));
    cudaMalloc((void**) &cscShadow.values, csc->capacity*sizeof(float));
    cudaMemcpy(cscShadow.colPtrs, csc->colPtrs, (csc->numCols + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cscShadow.rowIdxs, csc->rowIdxs, csc->capacity*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cscShadow.values, csc->values, csc->capacity*sizeof(float), cudaMemcpyHostToDevice);
    CSCMatrix* csc_d;
    cudaMalloc((void**) &csc_d, sizeof(CSCMatrix));
    cudaMemcpy(csc_d, &cscShadow, sizeof(CSCMatrix), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return csc_d;
}

void copyCSRfromGPU(CSRMatrix* csr_d, CSRMatrix* csr) {
    CSRMatrix csrShadow;
    cudaMemcpy(&csrShadow, csr_d, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    assert(csr->numRows == csrShadow.numRows);
    assert(csr->numCols == csrShadow.numCols);
    assert(csr->capacity >= csrShadow.nnz);
    csr->nnz = csrShadow.nnz;
    cudaMemcpy(csr->rowPtrs, csrShadow.rowPtrs, csrShadow.nnz*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(csr->colIdxs, csrShadow.colIdxs, csrShadow.nnz*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(csr->values, csrShadow.values, csrShadow.nnz*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void sparseNN(Vector* result, COOMatrix* featureVectors, COOMatrix** layerWeights, float bias, unsigned int numLayers) {


	
	Timer timer;

	// Convert featureVectors to CSR
    startTime(&timer);
    CSRMatrix* Y0 = createEmptyCSR(featureVectors->numRows, featureVectors->numCols, CAPACITY); 
    convertCOOtoCSR(featureVectors, Y0);
    CSRMatrix* Y0_d = createEmptyCSR_d(featureVectors->numRows, featureVectors->numCols, CAPACITY); 

    stopTimeAndPrint(&timer, "Convert feature vectors to CSR");

    // Convert layer weights to CSC
    startTime(&timer);
    CSCMatrix* W[numLayers];
    CSCMatrix* W_d[numLayers];
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        W[layer] = createCSCfromCOO(layerWeights[layer]);
        W_d[layer] = createCSCfromCSC_d(W[layer]);
    }
	stopTimeAndPrint(&timer, "Convert weights to CSR");

	
	// Temporary buffer
	startTime(&timer);
	COOMatrix *tmp = createEmptyCOO(Y0->numRows, Y0->numCols, Y0->capacity);
	COOMatrix *tmp_d = createEmptyCOO_d(Y0->numRows, Y0->numCols, Y0->capacity);
	stopTimeAndPrint(&timer, "Allocate temporary buffer");

	// Loop over layers
	CSRMatrix *Yin = Y0;
	COOMatrix *Yout = tmp;
	CSRMatrix *Yin_d = Y0_d;
	COOMatrix *Yout_d = tmp_d;


	


	copyCSRtoGPU(Yin, Yin_d);
	//kernel loop
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Loop over layers
	for (unsigned int layer = 0; layer < numLayers; ++layer) {

		// SpMSpM
		printf("Computing layer %u (SpMSpM)", layer);
		startTime(&timer);

        cudaMemset(&Yout_d->nnz, 0, sizeof(unsigned int));

		dim3 numThreadsPerBlock3(threads, threads);
		dim3 numBlocks3((W_d[layer].numCols + numThreadsPerBlock3.x - 1) / numThreadsPerBlock3.x, (Yin->numRows + numThreadsPerBlock3.y - 1) / numThreadsPerBlock3.y);

		spmspm << <numBlocks3, numThreadsPerBlock3 >> > (Yout_d, Yin_d, W_d[layer],bias);

		cudaDeviceSynchronize();

		// inBuffer_d.nnz = *out_nnz_h;
		// inBuffer_d.numCols = W_d[layer].numCols;

		printf("kernel time for layer %u", layer);
		stopTimeAndPrint(&timer, "");

		startTime(&timer);

		// Convert COO to CSR
		startTime(&timer);
		convertCOOtoCSR_d(Yout_d, Yin_d);
		stopTimeAndPrint(&timer, "    Converting COO to CSR device");

		cudaDeviceSynchronize();


	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	// Copy data from GPU
	startTime(&timer);
	copyCSRfromGPU(Yin_d,Yin);
	cudaDeviceSynchronize();
	stopTime(&timer);
	printElapsedTime(timer, "Copy from GPU time");


	//nonzero rows
	startTime(&timer);
	findNonzeroRows(result, Yin);
	stopTimeAndPrint(&timer, "Find nonzero rows");

	for (unsigned int layer = 0; layer < numLayers; ++layer) {
		cudaFree(W_d[layer].colPtrs);
		cudaFree(W_d[layer].rowIdxs);
		cudaFree(W_d[layer].values);

	}


	cudaDeviceSynchronize();
	stopTime(&timer);

	printElapsedTime(timer, "Deallocation time");

	// Free buffers
	startTime(&timer);
	freeCSR(Y0);
	for (unsigned int layer = 0; layer < numLayers; ++layer) {
		freeCSC(W[layer]);
	}
    freeCOO(tmp);
	stopTimeAndPrint(&timer, "Deallocate memory");


}
