#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"

#define THRESHOLD 0.000001
#define YMAX 32
#define threads 32

__global__ void spmspm(COOMatrix *result, unsigned int* nnz_out, CSRMatrix A, CSCMatrix B, float bias){ 
	unsigned int r= blockIdx.y*blockDim.y +threadIdx.y;
    	unsigned int c= blockIdx.x*blockDim.x + threadIdx.x;
	if(r < A.numRows && c < B.numCols){
		unsigned int rowPtrA = A.rowPtrs[0];
		unsigned int nnzA = A.rowPtrs[r + 1] - rowPtrA;
		if(nnzA>0) {
			unsigned int* colIdxsA = A.colIdxs + rowPtrA;
                        float* valueA = A.values + rowPtrA;
			unsigned int colPtrB = B.colPtrs[0];
			unsigned int nnzB = B.colPtrs[c + 1] - colPtrB;
			if(nnzB>0) {
				unsigned int* rowIdxsB = B.rowIdxs + colPtrB;
                                float* valueB = B.values + colPtrB;
                                float sum = 0.0f;
                                unsigned int ia = 0, ib = 0;
				while(ia < nnzA && ib < nnzB) { 
					unsigned int colIdx = colIdxsA[ia];
                                        unsigned int rowIdx = rowIdxsB[ib];
					if(colIdx < rowIdx) {
                                                ia++;
                                        } else if(colIdx > rowIdx) {
                                                ib++;
                                        }
					++ia;
					++ib;
				}
				*nnz_out = r;
				result->values[0] = c;
				
			}
		}
	}
}

COOMatrix* createEmptyCOO(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
        COOMatrix *coo = (COOMatrix *)malloc(sizeof(COOMatrix));
        coo->rowIdxs= (unsigned int *)malloc(capacity * sizeof(unsigned int));
        coo->colIdxs= (unsigned int *)malloc(capacity * sizeof(unsigned int));
        coo->values= (float *)malloc( capacity * sizeof(float));
        coo->numRows = numRows;
        coo->numCols = numCols;
        coo->nnz = 0;
        coo->capacity = capacity;
        return coo;
}

COOMatrix* sortCOO(COOMatrix *A){
         for (unsigned int i = 0; i < A->nnz; i++)
                for (unsigned int j = 0; j < A->nnz-i-1; j++)
                {    if (A->rowIdxs[j] > A->rowIdxs[j+1]){
                                unsigned int r = A->rowIdxs[j];
                                unsigned int c =  A->colIdxs[j];
                                float v = A->values[j];
                                A->rowIdxs[j] = A->rowIdxs[j+1];
                                A->colIdxs[j] = A->colIdxs[j+1];
                                A->values[j] = A->values[j+1];
                                A->rowIdxs[j+1] = r;
                                A->colIdxs[j+1] = c;
                                A->values[j+1] = v;
                        }
                }
         int begin = 0;
         for(unsigned int i  = 0 ;  i < A->nnz -1 ; i++)
         {
                 if(A->rowIdxs[i] != A->rowIdxs[i+1])
                 {
                        for(int k = begin ;  k< i + begin; k++)
                                for (int m = begin ; m < i + begin - k -1 ;m++)
                                        if(A->colIdxs[m] > A->colIdxs[m+1]){
                                                unsigned int c = A->colIdxs[m];
                                                float v = A->values[m];
                                                A->colIdxs[m] = A->colIdxs[m+1];
                                                A->values[m] = A->values[m+1];
                                                A->colIdxs[m+1] =c;
                                                A->values[m+1] = v;
                                        }
                        begin= i+1;
                }
        }
        return A;
 }
void sparseNN(Vector* result, COOMatrix* featureVectors, COOMatrix** layerWeights, float bias, unsigned int numLayers) {
	Timer timer;
	CSRMatrix* Y0 = createCSRfromCOO(featureVectors);
	CSCMatrix* W[numLayers];
	for (unsigned int layer = 0; layer < numLayers; ++layer) {
			W[layer] = createCSCfromCOO(layerWeights[layer]);
	}
	COOMatrix *outBuffer = createEmptyCOO(Y0->numRows, Y0->numCols, 5 * Y0->nnz);
    //outBuffer_d allocation
	COOMatrix *outBuffer_d; 
	unsigned int* out_rowIdxs_d;
	unsigned int* out_colIdxs_d;
	float* out_values_d;
	unsigned int* out_nnz_d;
	unsigned int* out_nnz_h = (unsigned int*) malloc(sizeof(unsigned int*));
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
	cudaMemcpy(out_nnz_d, &(outBuffer->nnz), sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(&(outBuffer_d->rowIdxs), &out_rowIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(outBuffer_d->colIdxs), &out_colIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(outBuffer_d->values), &out_values_d, sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(&out_nnz_d, &out_nnz_h, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	
	
	//Weights
	CSCMatrix W_d[numLayers];
        for (unsigned int layer = 0; layer < numLayers; ++layer) {
                W_d[layer].numRows = W[layer]->numRows;
                W_d[layer].numCols = W[layer]->numCols;
                W_d[layer].nnz = W[layer]->nnz;
                W_d[layer].capacity = W[layer]->capacity;
                cudaMalloc((void**)&W_d[layer].colPtrs, W[layer]->numCols * sizeof(unsigned int));
                cudaMalloc((void**)&W_d[layer].rowIdxs, W[layer]->numRows * sizeof(unsigned int));
                cudaMalloc((void**)&W_d[layer].values, W[layer]->numRows * sizeof(float));
        }

        for (unsigned int layer = 0; layer < numLayers; ++layer) {
                cudaMemcpy(W_d[layer].colPtrs, W[layer]->colPtrs, W[layer]->numCols * sizeof(unsigned int), cudaMemcpyHostToDevice);
                cudaMemcpy(W_d[layer].rowIdxs, W[layer]->rowIdxs, W[layer]->numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
                cudaMemcpy(W_d[layer].values, W[layer]->values, W[layer]->numRows * sizeof(float), cudaMemcpyHostToDevice);
	}
	
	CSRMatrix *inBuffer = Y0;
	CSRMatrix inBuffer_d;
        inBuffer_d.numRows = inBuffer->numRows;
        inBuffer_d.numCols = inBuffer->numCols;
        inBuffer_d.nnz = inBuffer->nnz;
        inBuffer_d.capacity = inBuffer->capacity;
        cudaMalloc((void**)&inBuffer_d.rowPtrs, inBuffer->numRows * sizeof(unsigned int));
        cudaMalloc((void**)&inBuffer_d.colIdxs, inBuffer->numCols * sizeof(unsigned int));
        cudaMalloc((void**)&inBuffer_d.values, inBuffer->numCols * sizeof(float));
	
	cudaMemcpy(inBuffer_d.rowPtrs, inBuffer->rowPtrs, inBuffer->numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(inBuffer_d.colIdxs, inBuffer->colIdxs, inBuffer->numCols * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(inBuffer_d.values, inBuffer->values, inBuffer->numCols * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaDeviceSynchronize();
	for (unsigned int layer = 0; layer < numLayers; ++layer) {
		printf("nnz before kernel call %d \n", outBuffer->nnz);
		printf("Computing layer %u (SpMSpM)", layer);
                startTime(&timer);
		dim3 numThreadsPerBlock(threads, threads);
        	dim3 numBlocks((W[layer]->numCols + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,(inBuffer->numRows + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
		spmspm <<<numBlocks, numThreadsPerBlock>>> (outBuffer_d, out_nnz_d, inBuffer_d, W_d[layer], bias);
		cudaDeviceSynchronize();
		stopTimeAndPrint(&timer, "");
		//copy back       
		cudaMemcpy(outBuffer->rowIdxs, out_rowIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(outBuffer->colIdxs, out_colIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(outBuffer->values, out_values_d, outBuffer->capacity * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(out_nnz_h, out_nnz_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		outBuffer->nnz = *out_nnz_h;

		cudaDeviceSynchronize();
		printf("nnzA %f \n", outBuffer->values[0]);
		//printf("cpu value at 0 %f \n", W[layer]->values[0]);
		printf("nnzB %d \n", outBuffer->nnz);
	}



}
