#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"


__global__ void spmspm(COOMatrix *result, unsigned int* nnz_out, CSCMatrix B){ 
	
	result->rowIdxs[0] = 1;
	result->colIdxs[0] = 1;
	result->values[0] = B.numCols;
	result->nnz = 10;
	*nnz_out = 42;
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
void sparseNN(Vector* result, COOMatrix* featureVectors, COOMatrix** layerWeights, float bias, unsigned int numLayers) {
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
	
	cudaMemcpy(inBuffer_d.rowPtrs, inBuffer->numRows, inBuffer->numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(inBuffer_d.colIdxs, inBuffer->numCols, inBuffer->numCols * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(inBuffer_d.values, inBuffer->numCols, inBuffer->numCols * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaDeviceSynchronize();
	printf("nnz before kernel call %d \n", outBuffer->nnz);

	spmspm <<<1, 1>>> (outBuffer_d, out_nnz_d, W_d[0]);
	cudaDeviceSynchronize();

	//copy back       
	cudaMemcpy(outBuffer->rowIdxs, out_rowIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(outBuffer->colIdxs, out_colIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(outBuffer->values, out_values_d, outBuffer->capacity * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&(outBuffer->nnz), out_nnz_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(out_nnz_h, out_nnz_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	printf("%f \n", outBuffer->values[0]);
	printf("nnz after kernel call %d \n", *out_nnz_h);



}
