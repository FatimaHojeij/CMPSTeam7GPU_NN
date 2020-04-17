/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>*/


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

__global__ void spmspm(COOMatrix *result, CSRMatrix A, CSCMatrix B, float bias, unsigned int* nnz_out) {
	unsigned int r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int c = blockIdx.x*blockDim.x + threadIdx.x;


	if (r < A.numRows && c < B.numCols) {
		unsigned int rowPtrA = A.rowPtrs[r];
		unsigned int nnzA = A.rowPtrs[r + 1] - rowPtrA;

		unsigned int colPtrB = B.colPtrs[c];
		unsigned int nnzB = B.colPtrs[c + 1] - colPtrB;
		if (nnzA > 0 && nnzB > 0) { // if a row is not all zeros , we do computation otherwise we skip row
				//ptrs to cols and vals of A[r]
				//unsigned int* colIdxsA = A.colIdxs + rowPtrA;
				//float* valueA = A.values + rowPtrA;
				//we will take one column of B

				 // if a col in B is not all zeros, we do computation otherwise skip//ptrs to rows and vals of B[c]
						//unsigned int* rowIdxsB = B.rowIdxs[colPtrB];
						//float* valueB = B.values[colPtrB];
						// Loop and find intersection
			float sum = 0.0f;
			unsigned int ia = 0, ib = 0;
			while (ia < nnzA && ib < nnzB) { // loops over all non zeros from A and B and stop when there is no more non zero

				unsigned int colIdx = A.colIdxs[rowPtrA + ia]; //single item col index from A
				unsigned int rowIdx = B.rowIdxs[colPtrB + ib]; //single item row index from B
				if (rowIdx < B.nnz && colIdx < A.nnz) {
					if (colIdx < rowIdx) {
						ia++;
					}
					else if (colIdx > rowIdx) {
						ib++;
					}
					else {
						sum += A.values[rowPtrA + ia] * B.values[ib + colPtrB];// do the multiplication of the row that matches the column
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
					unsigned int nnzIndxTemp = atomicAdd(nnz_out, 1); //counts how many non zero elements I have
					result->rowIdxs[nnzIndxTemp] = r;
					result->colIdxs[nnzIndxTemp] = c;
					result->values[nnzIndxTemp] = sum;
				}
			}


		}
	}

}


//extern __shared__ unsigned int array[];
__global__ void histogram_private_kernel(unsigned int* rowIdxs, unsigned int* rowPtrs, unsigned int nnz, unsigned int numRows) {


	unsigned int t = blockDim.x*blockIdx.x + threadIdx.x;

	// if (t < numRows + 1) {
	// 	rowPtrs[t] = 0;
	// }

	// __syncthreads();

	if (t < nnz) {
		unsigned int rIdx = rowIdxs[t];
		atomicAdd(&rowPtrs[rIdx], 1);
	}



}


__global__ void scan_kernel(unsigned int* input, unsigned int* output, unsigned int* partialSums, unsigned int N) {

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
		output[i] = input_s[tid];
	}
	if (i + BLOCK_DIM < N)
	{
		output[i + BLOCK_DIM] = input_s[tid + BLOCK_DIM];
	}

}

__global__ void add_kernel(unsigned int* output, unsigned int* partialSums, unsigned int N) {

	// TODO
	unsigned int i = 2 * blockIdx.x*blockDim.x + threadIdx.x;
	if (blockIdx.x != 0) {
		if (i < N) {
			output[i] += partialSums[blockIdx.x];
		}
		if (i + BLOCK_DIM < N) {
			output[i + BLOCK_DIM] += partialSums[blockIdx.x];
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



__global__ void Binning_kernel(unsigned int* inrowIdxs, unsigned int* incolIdxs, float* invalues, unsigned int* rowPtrs, unsigned int* colIdxs, float* values, unsigned int nnz, unsigned int numRows, unsigned int* rowPtrsBin) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;


	// if (i < numRows + 1) {
	// 	rowPtrsBin[i] = 0;
	// }

	// __syncthreads();

	if (i < nnz) {
		unsigned int row = inrowIdxs[i];
		unsigned int col = incolIdxs[i];
		float val = invalues[i];
		unsigned int init = rowPtrs[row];
		unsigned int nnzIdx = atomicAdd(&rowPtrsBin[row], 1);
		colIdxs[nnzIdx+init] = col;
		values[nnzIdx+init]=val;
	}


}

__global__ void  sorting_kernel( unsigned int* colIdxs, float* values,unsigned int* rowPtrs, unsigned int numRows){
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	if (i < numRows) {
		unsigned int rowPtrA = rowPtrs[i];
		unsigned int nnzA = (rowPtrs[i + 1] - rowPtrs[i]);
		if(nnzA>0)
			for (unsigned int j = rowPtrA; j < rowPtrA + nnzA - 1;++j) {

				for (unsigned int k = rowPtrA; k < rowPtrA + nnzA - j - 1; ++k) {
					if (colIdxs[k] > colIdxs[k + 1]) {
						//swap col
						unsigned int tmp = colIdxs[k];
						colIdxs[k] = colIdxs[k + 1];
						colIdxs[k + 1] = tmp;
						//swap float
						float valtmp = values[k];
						values[k] = values[k + 1];
						values[k + 1] = valtmp;
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

COOMatrix* createEmptyCOO(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
	COOMatrix *coo = (COOMatrix *)malloc(sizeof(COOMatrix));
	coo->rowIdxs = (unsigned int *)malloc(capacity * sizeof(unsigned int));
	coo->colIdxs = (unsigned int *)malloc(capacity * sizeof(unsigned int));
	coo->values = (float *)malloc(capacity * sizeof(float));
	coo->numRows = numRows;
	coo->numCols = numCols;
	coo->nnz = 0;
	coo->capacity = CAPACITY;
	for (unsigned int i = 0; i < coo->capacity;++i) {
		coo->rowIdxs[i] = 0;
		coo->colIdxs[i] = 0;
		coo->values[i] = 0.0f;
	}
	return coo;
}

void sparseNN(Vector* result, COOMatrix* featureVectors, COOMatrix** layerWeights, float bias, unsigned int numLayers) {
	//const unsigned int _numLayers = 120;

	Timer timer;

	// Convert featureVectors to CSR
	startTime(&timer);
	CSRMatrix* Y0 = createCSRfromCOO(featureVectors);
	stopTimeAndPrint(&timer, "Convert feature vectors to CSR");

	// Convert layer weights to CSC
	startTime(&timer);
	CSCMatrix* W[numLayers];
	//CSCMatrix* W[_numLayers];
	for (unsigned int layer = 0; layer < numLayers; ++layer) {
		W[layer] = createCSCfromCOO(layerWeights[layer]);
	}
	stopTimeAndPrint(&timer, "Convert weights to CSC");

	// Double buffers
	startTime(&timer);
	COOMatrix *tmp = createEmptyCOO(Y0->numRows, Y0->numCols, CAPACITY);
	CSRMatrix *inBuffer = Y0;
	COOMatrix *outBuffer = tmp;
	stopTimeAndPrint(&timer, "Allocate temporary buffer");

	inBuffer->capacity = CAPACITY;

	// Allocate GPU memory
	startTime(&timer);

	outBuffer->capacity = CAPACITY;

	//allocating inbuffer address and value
	CSRMatrix inBuffer_d;
	//CSRMatrix* inBuffer_d;
	inBuffer_d.numRows = inBuffer->numRows;
	inBuffer_d.numCols = inBuffer->numCols;
	inBuffer_d.nnz = inBuffer->nnz;
	inBuffer_d.capacity = inBuffer->capacity;
	cudaMalloc((void**)&inBuffer_d.rowPtrs, (inBuffer->numRows + 1) * sizeof(unsigned int));
	cudaMalloc((void**)&inBuffer_d.colIdxs, CAPACITY * sizeof(unsigned int));
	cudaMalloc((void**)&inBuffer_d.values, CAPACITY * sizeof(float));

	cudaMemcpy(inBuffer_d.rowPtrs, inBuffer->rowPtrs, (inBuffer->numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(inBuffer_d.colIdxs, inBuffer->colIdxs, (inBuffer->nnz) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(inBuffer_d.values, inBuffer->values, inBuffer->nnz * sizeof(float), cudaMemcpyHostToDevice);

	//cudaMalloc(&inBuffer_d, sizeof(CSRMatrix));

	//cudaMemcpy(inBuffer_d,&inBuffer_d,sizeof(CSRMatrix),cudaMemcpyHostToDevice);

	printf("inbuffer allocated\n");

	/////////////////////////

	//outBuffer_d allocation
	COOMatrix *outBuffer_d;
	unsigned int* out_rowIdxs_d;
	unsigned int* out_colIdxs_d;
	float* out_values_d;
	unsigned int* out_nnz_d;
	unsigned int* out_nnz_h = (unsigned int*)malloc(sizeof(unsigned int*));
	*out_nnz_h = outBuffer->nnz;
	cudaMalloc((void**)&outBuffer_d, sizeof(COOMatrix));
	cudaMalloc((void**)&out_rowIdxs_d, outBuffer->capacity * sizeof(unsigned int));
	cudaMalloc((void**)&out_colIdxs_d, outBuffer->capacity * sizeof(unsigned int));
	cudaMalloc((void**)&out_values_d, outBuffer->capacity * sizeof(float));
	cudaMalloc((void**)&out_nnz_d, sizeof(unsigned int));

	//copying outbuffer
	cudaMemcpy(outBuffer_d, outBuffer, sizeof(COOMatrix), cudaMemcpyHostToDevice);
	cudaMemcpy(out_rowIdxs_d, outBuffer->rowIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(out_colIdxs_d, outBuffer->colIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(out_values_d, outBuffer->values, outBuffer->capacity * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(out_nnz_d, out_nnz_h, sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(&(outBuffer_d->rowIdxs), &out_rowIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(outBuffer_d->colIdxs), &out_colIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(outBuffer_d->values), &out_values_d, sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(outBuffer_d->numRows), &(outBuffer->numRows), sizeof(unsigned int), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();



	printf("outbuffer allocated\n");
	//////////////////////////////////


	// allocating W_d
	//CSCMatrix W_d[_numLayers];
	CSCMatrix W_d[numLayers];
	for (unsigned int layer = 0; layer < numLayers; ++layer) {
		W_d[layer].numRows = W[layer]->numRows;
		W_d[layer].numCols = W[layer]->numCols;
		W_d[layer].nnz = W[layer]->nnz;
		W_d[layer].capacity = W[layer]->capacity;
		cudaMalloc((void**)&W_d[layer].colPtrs, (W[layer]->numCols + 1) * sizeof(unsigned int));
		cudaMalloc((void**)&W_d[layer].rowIdxs, W[layer]->capacity * sizeof(unsigned int));
		cudaMalloc((void**)&W_d[layer].values, W[layer]->capacity * sizeof(float));
	}

	for (unsigned int layer = 0; layer < numLayers; ++layer) {
		cudaMemcpy(W_d[layer].colPtrs, W[layer]->colPtrs, (W[layer]->numCols + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(W_d[layer].rowIdxs, W[layer]->rowIdxs, W[layer]->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(W_d[layer].values, W[layer]->values, W[layer]->capacity * sizeof(float), cudaMemcpyHostToDevice);
	}

	cudaDeviceSynchronize();
	stopTime(&timer);
	printElapsedTime(timer, "Allocation and copy time on GPU Memory");



	//unsigned int uMax = (unsigned int)~0;
	//cudaMemcpyToSymbol(&u_Max, &uMax, sizeof(unsigned int));

	unsigned int  *rowPtrstmp_d;
	//rowPtrstmp = (unsigned int *)malloc((inBuffer_d.numRows + 1) * sizeof(unsigned int));
	unsigned int *rowPtrstmp;
	rowPtrstmp = (unsigned int *)malloc((inBuffer_d.numRows + 1) * sizeof(unsigned int));

	//memset(rowPtrstmp, 0, sizeof (unsigned int) * (inBuffer_d.numRows + 1));


	cudaMalloc((void**)&rowPtrstmp_d, (inBuffer_d.numRows + 1) * sizeof(unsigned int));

	for(unsigned int i=0; i<inBuffer_d.numRows+1;i++){
	 	rowPtrstmp[i]=0;
	}

	cudaMemcpy(rowPtrstmp_d, rowPtrstmp, (inBuffer_d.numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);

	//kernel loop
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Loop over layers
	for (unsigned int layer = 0; layer < numLayers; ++layer) {

		// cudaMemcpy(out_nnz_h, out_nnz_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		// printf(" before nnz %d\n", *out_nnz_h);
		// if(layer!=0){
		// 	cudaMemcpy(inBuffer->rowPtrs, inBuffer_d.rowPtrs, (inBuffer_d.numRows + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		// 	cudaMemcpy(inBuffer->colIdxs, inBuffer_d.colIdxs, inBuffer_d.nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		// 	cudaMemcpy(inBuffer->values, inBuffer_d.values, inBuffer_d.nnz * sizeof(float), cudaMemcpyDeviceToHost);

		// 	printf("Layer %d \n",layer)
		// 	for(int i=0; i<inBuffer_d.numRows+1;++i){

		// 		unsigned int rowPtr = inBuffer->row[i];
		// 		unsigned int nnzr = inBuffer->row[i+1]-inBuffer->row[i+1];

		// 		for(unsigned int j =rowPtr;j<rowPtr+nnzr;++j){
		// 			printf("row %u , col %u, val %f\n",i,inBuffer->colIdxs[j],inBuffer->values[j])
		// 		}
		// 	}
		// }	

		// SpMSpM
		printf("Computing layer %u (SpMSpM)", layer);
		startTime(&timer);



		dim3 numThreadsPerBlock3(threads, threads);
		dim3 numBlocks3((W_d[layer].numCols + numThreadsPerBlock3.x - 1) / numThreadsPerBlock3.x, (inBuffer_d.numRows + numThreadsPerBlock3.y - 1) / numThreadsPerBlock3.y);

		spmspm << <numBlocks3, numThreadsPerBlock3 >> > (outBuffer_d, inBuffer_d, W_d[layer], bias, out_nnz_d);

		cudaDeviceSynchronize();



		stopTimeAndPrint(&timer, "");

		cudaMemcpy(out_nnz_h, out_nnz_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		printf("nnz %d\n", *out_nnz_h);


		inBuffer_d.nnz = *out_nnz_h;
		inBuffer_d.numCols = W_d[layer].numCols;
		inBuffer->numCols = inBuffer_d.numCols;
		inBuffer->numRows = inBuffer_d.numRows;
		inBuffer->nnz = inBuffer_d.nnz;

		cudaDeviceSynchronize();

		printf("kernel time for layer %u", layer);
		stopTimeAndPrint(&timer, "");

		startTime(&timer);
		//calling histogram to fill rowPtrs of inBuffer
		unsigned int numThreadsPerBlock = 1024;
		unsigned int numBlocks = (*out_nnz_h + numThreadsPerBlock - 1) / numThreadsPerBlock;


		// cudaMemcpy(outBuffer->rowIdxs, out_rowIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		// cudaMemcpy(out_rowIdxs_d, outBuffer->rowIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
		if(layer!=0)
			cudaMemcpy(rowPtrstmp_d, rowPtrstmp, (inBuffer_d.numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
		
		histogram_private_kernel << < numBlocks, numThreadsPerBlock >> > (out_rowIdxs_d, rowPtrstmp_d, *out_nnz_h, inBuffer_d.numRows);

		cudaDeviceSynchronize();

		printf("Histogram time for layer %u", layer);
		stopTimeAndPrint(&timer, "");

		startTime(&timer);

		//calling the scan kernel to scan kernel ptrs
		const unsigned int numElementsPerBlock = 2 * numThreadsPerBlock;
		numBlocks = ((inBuffer_d.numRows + 1) + numElementsPerBlock - 1) / numElementsPerBlock;

		// Allocate partial sums
		unsigned int *partialSums_d;
		cudaMalloc((void**)&partialSums_d, numBlocks * sizeof(unsigned int));
		cudaDeviceSynchronize();

		// Call kernel
		scan_kernel << < numBlocks, numThreadsPerBlock >> > (rowPtrstmp_d, inBuffer_d.rowPtrs, partialSums_d, inBuffer_d.numRows + 1);

		cudaDeviceSynchronize();
		// Scan partial sums then add

		if (numBlocks > 1) {

			// Scan partial sums
			scan_gpu_d(partialSums_d, partialSums_d, numBlocks);

			// Add scanned sums
			add_kernel << < numBlocks, numThreadsPerBlock >> > (inBuffer_d.rowPtrs, partialSums_d, inBuffer_d.numRows + 1);

		}

		cudaDeviceSynchronize();
				

		   
		cudaMemcpy(rowPtrstmp, inBuffer_d.rowPtrs, sizeof(unsigned int) * (inBuffer_d.numRows + 1), cudaMemcpyDeviceToHost);

		//printf("test %u\n", rowPtrstmp[inBuffer_d.numRows]);

		// Free memory

		cudaFree(partialSums_d);

		printf("Scan time for layer %u", layer);
		stopTimeAndPrint(&timer, "");
		startTime(&timer);

		//Binning
		for(unsigned int i=0; i<inBuffer_d.numRows+1;i++){
			rowPtrstmp[i]=0;
		}

		cudaMemcpy(rowPtrstmp_d, rowPtrstmp, (inBuffer_d.numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		numBlocks = (*out_nnz_h + numThreadsPerBlock - 1) / numThreadsPerBlock;

		Binning_kernel << < numBlocks, numThreadsPerBlock >> > (out_rowIdxs_d, out_colIdxs_d, out_values_d, inBuffer_d.rowPtrs, inBuffer_d.colIdxs, inBuffer_d.values, *out_nnz_h, inBuffer_d.numRows,rowPtrstmp_d);

		cudaDeviceSynchronize();

		// cudaFree(rowPtrstmp_d);
		// free(rowPtrstmp);
		//Sorting
		numBlocks = ((inBuffer_d.numRows +1) + numThreadsPerBlock - 1) / numThreadsPerBlock;


		sorting_kernel <<< numBlocks, numThreadsPerBlock >>>(inBuffer_d.colIdxs, inBuffer_d.values, inBuffer_d.rowPtrs, inBuffer_d.numRows);
		// cudaMemcpy(outBuffer->colIdxs, inBuffer_d.colIdxs, inBuffer_d.capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);

		// cudaMemcpy(outBuffer->values, inBuffer_d.values, inBuffer_d.capacity * sizeof(float), cudaMemcpyDeviceToHost);

		FILE* f = fopen("./out_gpu_sorting.txt","w");
		//cudaMemcpy(inBuffer->rowPtrs, inBuffer_d.rowPtrs, (inBuffer_d.numRows + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(inBuffer->colIdxs, inBuffer_d.colIdxs, inBuffer_d.nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(inBuffer->values, inBuffer_d.values, inBuffer_d.nnz * sizeof(float), cudaMemcpyDeviceToHost);

		//printf("Layer %d \n",layer)
		for(int i=0; i<inBuffer_d.nnz;++i){
			
			fprintf(f,"col %u  val %f\n",inBuffer->colIdxs[i],inBuffer->values[i]);

		}
		fclose(f);
		break;
		cudaDeviceSynchronize();

		//empty the outbuffer
		printf("Converting time for layer %u", layer);
		stopTimeAndPrint(&timer, "");

		// cudaMemcpy(outBuffer_d, outBuffer, sizeof(COOMatrix), cudaMemcpyHostToDevice);
		// cudaMemcpy(out_rowIdxs_d, outBuffer->rowIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
		// cudaMemcpy(out_colIdxs_d, outBuffer->colIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
		// cudaMemcpy(out_values_d, outBuffer->values, outBuffer->capacity * sizeof(float), cudaMemcpyHostToDevice);
		*out_nnz_h =0;
		cudaMemcpy(out_nnz_d, out_nnz_h, sizeof(unsigned int), cudaMemcpyHostToDevice);
		// cudaMemcpy(&(outBuffer_d->rowIdxs), &out_rowIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
		// cudaMemcpy(&(outBuffer_d->colIdxs), &out_colIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
		// cudaMemcpy(&(outBuffer_d->values), &out_values_d, sizeof(float*), cudaMemcpyHostToDevice);
		//cudaMemcpy(&(outBuffer_d->numRows), &(outBuffer->numRows), sizeof(unsigned int), cudaMemcpyHostToDevice);


		for(unsigned int i=0; i<inBuffer_d.numRows+1;i++){
			rowPtrstmp[i]=0;
	   	}

		cudaDeviceSynchronize();

		
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	// Copy data from GPU
	startTime(&timer);
	return;

	//TODO
	inBuffer->numRows = inBuffer_d.numRows;
	inBuffer->numCols = inBuffer_d.numCols;
	inBuffer->nnz = inBuffer_d.nnz;
	cudaMemcpy(inBuffer->rowPtrs, inBuffer_d.rowPtrs, (inBuffer_d.numRows + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(inBuffer->colIdxs, inBuffer_d.colIdxs, inBuffer_d.nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(inBuffer->values, inBuffer_d.values, inBuffer_d.nnz * sizeof(float), cudaMemcpyDeviceToHost);


	cudaDeviceSynchronize();
	stopTime(&timer);
	printElapsedTime(timer, "Copy from GPU time");

	//CPU
	// Find 
	//nonzero rows
	startTime(&timer);
	findNonzeroRows(result, inBuffer);
	stopTimeAndPrint(&timer, "Find nonzero rows");

	// Free GPU memory
	startTime(&timer);
	cudaFree(inBuffer_d.rowPtrs);
	cudaFree(inBuffer_d.colIdxs);
	cudaFree(inBuffer_d.values);
	cudaFree(outBuffer_d);


	// cudaFree(tmpOutBuffer.rowIdxs);
	// cudaFree(tmpOutBuffer.colIdxs);
	// cudaFree(tmpOutBuffer.values);

	//cudaFree(inBuffer_d);

	free(inBuffer);
	free(outBuffer);
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
	//freeCSR(Y0);
	for (unsigned int layer = 0; layer < numLayers; ++layer) {
		freeCSC(W[layer]);
	}

	stopTimeAndPrint(&timer, "Deallocate memory");


}
