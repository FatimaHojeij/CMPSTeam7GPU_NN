
// row+1; swapping; nnzidx; syncthreads
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"


#define THRESHOLD 0.000001
#define YMAX 32
#define threads 512


__global__ void spmspm(COOMatrix *result){ 
	
	result->rowIdxs[0] = 1;
	result->colIdxs[0] = 1;
	result->values[0] = 5;
}

void sparseNN(Vector* result, COOMatrix* outBuffer, COOMatrix** layerWeights, float bias, unsigned int numLayers) {

    //outBuffer_d allocation
    COOMatrix *outBuffer_d;
	unsigned int* out_rowIdxs_d;
	unsigned int* out_colIdxs_d;
	float* out_values_d;
	cudaMalloc((void**)&outBuffer_d, sizeof(COOMatrix));
	cudaMalloc((void**)&out_rowIdxs_d, outBuffer->capacity * sizeof(unsigned int));
	cudaMalloc((void**)&out_colIdxs_d, outBuffer->capacity * sizeof(unsigned int));
	cudaMalloc((void**)&out_values_d, outBuffer->capacity * sizeof(float));
        
        
 
	//copying outbuffer
	//cudaMemcpy(outBuffer_d, outBuffer, sizeof(COOMatrix), cudaMemcpyHostToDevice);
	cudaMemcpy(out_rowIdxs_d, outBuffer->rowIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(out_colIdxs_d, outBuffer->colIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(out_values_d, outBuffer->values, outBuffer->capacity * sizeof(float), cudaMemcpyHostToDevice);
	outBuffer->rowIdxs = out_rowIdxs_d;
	outBuffer->colIdxs = out_colIdxs_d;
	outBuffer->values = out_values_d;
	cudaMemcpy(outBuffer_d, outBuffer, sizeof(COOMatrix), cudaMemcpyHostToDevice);
	
    cudaDeviceSynchronize();

	spmspm <<<1, 1>>> (outBuffer_d);
		
	cudaDeviceSynchronize();			
           
	cudaMemcpy(outBuffer, outBuffer_d, sizeof(COOMatrix), cudaMemcpyDeviceToHost);
	cudaMemcpy(outBuffer->rowIdxs, out_rowIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(outBuffer->colIdxs, out_colIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(outBuffer->values, out_values_d, outBuffer->capacity * sizeof(float), cudaMemcpyDeviceToHost);
	
	// 9. Point to host pointer in host struct.
    //h_a->arr = h_arr;
		
	printf("%f \n", outBuffer->values[0]);


}
