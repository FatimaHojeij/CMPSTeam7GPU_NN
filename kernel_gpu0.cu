#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"


__global__ void spmspm(COOMatrix *result, unsigned int* nnz_out, CSRMatrix A){ 
	
	result->rowIdxs[0] = 1;
	result->colIdxs[0] = 1;
	result->values[0] = 5;
	*nnz_out = A->values[0];
}

void sparseNN(Vector* result, COOMatrix* outBuffer, COOMatrix** layerWeights, float bias, unsigned int numLayers) {

	CSCMatrix* W[numLayers];
	for (unsigned int layer = 0; layer < numLayers; ++layer) {
			W[layer] = createCSCfromCOO(layerWeights[layer]);
	}
	stopTimeAndPrint(&timer, "Convert weights to CSC");
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
	cudaDeviceSynchronize();
	printf("nnz before kernel call %d \n", outBuffer->nnz);

	unsigned int layer = 0;
	CSCMatrix* W_d;
		unsigned int* w_colPtrs_d;
		unsigned int* w_rowIdxs_d;
		float* w_values_d;
		cudaMalloc((void**)&W_d, sizeof(CSCMatrix));
        cudaMalloc((void**)&w_colPtrs_d, (W[layer]->numCols + 1)* sizeof(unsigned int));
        cudaMalloc((void**)&w_rowIdxs_d, W[layer]->numRows * sizeof(unsigned int));
        cudaMalloc((void**)&w_values_d, W[layer]->numRows * sizeof(float));
		//copying W_d[layer]
		cudaMemcpy(W_d, W[layer], sizeof(CSCMatrix), cudaMemcpyHostToDevice);
		cudaMemcpy(w_colPtrs_d, W[layer]->colPtrs, W[layer]->numCols * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(w_rowIdxs_d, W[layer]->rowIdxs, W[layer]->numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(w_values_d, W[layer]->values, W[layer]->numRows * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(&(W_d->colPtrs), &w_colPtrs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
		cudaMemcpy(&(W_d->rowIdxs), &w_rowIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
		cudaMemcpy(&(W_d->values), &w_values_d, sizeof(float*), cudaMemcpyHostToDevice);
		
	spmspm <<<1, 1>>> (outBuffer_d, out_nnz_d, *W_d);
	cudaDeviceSynchronize();

	//copy back       
	cudaMemcpy(outBuffer->rowIdxs, out_rowIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(outBuffer->colIdxs, out_colIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(outBuffer->values, out_values_d, outBuffer->capacity * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(out_nnz_h, out_nnz_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	outBuffer->nnz = *out_nnz_h;
	printf("%f \n", outBuffer->values[0]);
	printf("nnz after kernel call %d \n", outBuffer->nnz);



}
