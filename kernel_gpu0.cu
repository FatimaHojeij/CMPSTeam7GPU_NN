
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"


__global__ void spmspm(COOMatrix *result){//, unsigned int* nnz_out){ 
	
	result->rowIdxs[0] = 1;
	result->colIdxs[0] = 1;
	result->values[0] = 5;
	result->nnz = 10;
	//*nnz_out = 42;
}

void sparseNN(Vector* result, COOMatrix* outBuffer, COOMatrix** layerWeights, float bias, unsigned int numLayers) {

    //outBuffer_d allocation
	COOMatrix *outBuffer_d; 
	unsigned int* out_rowIdxs_d;
	unsigned int* out_colIdxs_d;
	float* out_values_d;
	outBuffer_d->nnz = outBuffer->nnz;
	//unsigned int* out_nnz_d;
	//unsigned int* out_nnz_h = (unsigned int*) malloc(sizeof(unsigned int*));
	//*out_nnz_h = outBuffer->nnz;
	cudaMalloc((void**)&outBuffer_d, sizeof(COOMatrix));
	cudaMalloc((void**)&out_rowIdxs_d, outBuffer->capacity * sizeof(unsigned int));
	cudaMalloc((void**)&out_colIdxs_d, outBuffer->capacity * sizeof(unsigned int));
	cudaMalloc((void**)&out_values_d, outBuffer->capacity * sizeof(float));
	//cudaMalloc((void**)&out_nnz_d, sizeof(unsigned int));



	//copying outbuffer
	cudaMemcpy(outBuffer_d, outBuffer, sizeof(COOMatrix), cudaMemcpyHostToDevice);
	cudaMemcpy(out_rowIdxs_d, outBuffer->rowIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(out_colIdxs_d, outBuffer->colIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(out_values_d, outBuffer->values, outBuffer->capacity * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(out_nnz_d, out_nnz_h, sizeof(unsigned int), cudaMemcpyHostToDevice);
	//cudaMemcpy(out_nnz_d, &(outBuffer->nnz), sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(&(outBuffer_d->rowIdxs), &out_rowIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(outBuffer_d->colIdxs), &out_colIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(outBuffer_d->values), &out_values_d, sizeof(float*), cudaMemcpyHostToDevice);
	//cudaMemcpy(&(outBuffer->nnz), &out_nnz_d, sizeof(unsigned int), cudaMemcpyHostToDevice);
	//cudaMemcpy(&out_nnz_d, &out_nnz_h, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	printf("nnz before kernel call %d \n", outBuffer->nnz);

	spmspm <<<1, 1>>> (outBuffer_d);//, out_nnz_d);
	cudaDeviceSynchronize();

	//copy back       
	cudaMemcpy(outBuffer->rowIdxs, out_rowIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(outBuffer->colIdxs, out_colIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(outBuffer->values, out_values_d, outBuffer->capacity * sizeof(float), cudaMemcpyDeviceToHost);
	outBuffer->nnz = outBuffer_d->nnz;
	//cudaMemcpy(outBuffer->nnz, &out_nnz_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&(outBuffer->nnz), out_nnz_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(out_nnz_h, out_nnz_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	printf("%f \n", outBuffer->values[0]);
	printf("nnz after kernel call %d \n", outBuffer->nnz);



}
