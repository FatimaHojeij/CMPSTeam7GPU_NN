// row+1; swapping; nnzidx; syncthreads
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"


#define THRESHOLD 0.000001
#define YMAX 32
#define threads 32


__global__ void spmspm(COOMatrix *result, CSRMatrix A, CSCMatrix B, float bias, unsigned int* nnz_out) {
    unsigned int r= blockIdx.y*blockDim.y +threadIdx.y;
    unsigned int c= blockIdx.x*blockDim.x + threadIdx.x;
    

        if(r < A.numRows && c < B.numCols){
                unsigned int rowPtrA = A.rowPtrs[r];
                unsigned int nnzA = A.rowPtrs[r + 1] - rowPtrA;
                if(nnzA>0) { // if a row is not all zeros , we do computation otherwise we skip row
                        //ptrs to cols and vals of A[r]
                        //unsigned int* colIdxsA = A.colIdxs + rowPtrA;
                        //float* valueA = A.values + rowPtrA;
                        //we will take one column of B
                        unsigned int colPtrB = B.colPtrs[c];
                        unsigned int nnzB = B.colPtrs[c + 1] - colPtrB;
                        if(nnzB>0) { // if a col in B is not all zeros, we do computation otherwise skip//ptrs to rows and vals of B[c]
                                //unsigned int* rowIdxsB = B.rowIdxs[colPtrB];
                                //float* valueB = B.values[colPtrB];
                                // Loop and find intersection
                                float sum = 0.0f;
                                unsigned int ia = 0, ib = 0;
                                while(ia < nnzA && ib < nnzB) { // loops over all non zeros from A and B and stop when there is no more non zero
                                        if((rowPtrA + ia)<A.nnz &&(colPtrB+ib)<B.nnz){
                                                unsigned int colIdx = A.colIdxs[rowPtrA + ia]; //single item col index from A
                                                unsigned int rowIdx = B.rowIdxs[colPtrB+ib]; //single item row index from B
                                                if(colIdx < rowIdx) {
                                                        ia++;
                                                } else if(colIdx > rowIdx) {
                                                        ib++;
                                                } else {
                                                        sum += A.values[rowPtrA + ia ]*B.values[ib+colPtrB];// do the multiplication of the row that matches the column
                                                        ia++;
                                                        ib++;
                                                }
                                        }
                                }
                                if(sum > THRESHOLD || sum < -THRESHOLD) { //if not smaller than abs(threshold)
                                        sum += bias; //add to it the bias
                                        //Remove negative and zero values
                                        if(sum > 0) {//if end result is positive otherwise I also do not want to add it to result
                                                if(sum>YMAX) { //make sure it is on an upper limit
                                                        sum = YMAX;
                                                }
                                                unsigned int nnzIndxTemp = atomicAdd(nnz_out,1); //counts how many non zero elements I have
                                                result->rowIdxs[nnzIndxTemp] = r;
                                                result->colIdxs[nnzIndxTemp] = c;
                                                result->values[nnzIndxTemp] = sum;
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

        Timer timer;

        // Convert featureVectors to CSR
        startTime(&timer);
 	CSRMatrix* Y0 = createCSRfromCOO(featureVectors);
	stopTimeAndPrint(&timer, "Convert feature vectors to CSR");

	// Convert layer weights to CSC
	startTime(&timer);
	CSCMatrix* W[numLayers];
	for (unsigned int layer = 0; layer < numLayers; ++layer) {
                W[layer] = createCSCfromCOO(layerWeights[layer]);
                if(layer==1 || layer==0){     
                        printf("layer %u\n",layer);   
                }
	}
	stopTimeAndPrint(&timer, "Convert weights to CSC");

	// Double buffers
	startTime(&timer);
	COOMatrix *tmp = createEmptyCOO(Y0->numRows, Y0->numCols, 5 * Y0->nnz);
	CSRMatrix *inBuffer = Y0;
	COOMatrix *outBuffer = tmp;
	stopTimeAndPrint(&timer, "Allocate temporary buffer");



        // Allocate GPU memory
        startTime(&timer);

        //result_d allocation
        //Vector *result_d;
        //result_d->nnz = result->nnz;
        //result_d->capacity = result->capacity;
        //cudaMalloc((void**)&result_d->data, result->capacity * sizeof(unsigned int));
        //inBuffer_d allocation
        
        
        //allocating inbuffer address and value
        CSRMatrix tmpInBuffer;
        CSRMatrix* inBuffer_d;
        tmpInBuffer.numRows = inBuffer->numRows;
        tmpInBuffer.numCols = inBuffer->numCols;
        tmpInBuffer.nnz = inBuffer->nnz;
        tmpInBuffer.capacity = inBuffer->capacity;
        cudaMalloc((void**)&tmpInBuffer.rowPtrs, (inBuffer->numRows + 1) * sizeof(unsigned int));
        cudaMalloc((void**)&tmpInBuffer.colIdxs, inBuffer->numCols * sizeof(unsigned int));
        cudaMalloc((void**)&tmpInBuffer.values, inBuffer->numCols * sizeof(float));

        cudaMemcpy(tmpInBuffer.rowPtrs, inBuffer->rowPtrs, inBuffer->numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(tmpInBuffer.colIdxs, inBuffer->colIdxs, inBuffer->numCols * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(tmpInBuffer.values, inBuffer->values, inBuffer->numCols * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&inBuffer_d, sizeof(CSRMatrix));

        cudaMemcpy(inBuffer_d,&tmpInBuffer,sizeof(CSRMatrix),cudaMemcpyHostToDevice);

        printf("inbuffer allocated\n");

        /////////////////////////

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
	//cudaMemcpy(&out_nnz_d, &out_nnz_h, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
        
        
        
        printf("outbuffer allocated\n");
        //////////////////////////////////

        
        // allocating W_d
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

        cudaDeviceSynchronize();
        stopTime(&timer);
        printElapsedTime(timer, "Allocation and copy time on GPU Memory");


        //kernel loop
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Loop over layers
        for (unsigned int layer = 0; layer < numLayers; ++layer) {

                // SpMSpM
                printf("Computing layer %u (SpMSpM)", layer);
                startTime(&timer);
  
                dim3 numThreadsPerBlock(threads, threads);
                dim3 numBlocks((W_d[layer].numCols + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,(inBuffer->numRows + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
                cudaMemset(out_nnz_d,0,sizeof(unsigned int));
                spmspm <<<numBlocks, numThreadsPerBlock>>> (outBuffer_d, tmpInBuffer, W_d[layer], bias,out_nnz_d);


                cudaDeviceSynchronize();

                cudaMemcpy(outBuffer->rowIdxs, out_rowIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                cudaMemcpy(outBuffer->colIdxs, out_colIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                cudaMemcpy(outBuffer->values, out_values_d, outBuffer->capacity * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(out_nnz_h, out_nnz_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                outBuffer->nnz = *out_nnz_h;
                printf("nnz %d\n", outBuffer->nnz);
                // for(int i =0; i<outBuffer->nnz;++i){
                //         printf(" i = %d, row = %d, col = %d\n", i,outBuffer->rowIdxs[i],outBuffer->colIdxs[i]);
                // }

                cudaDeviceSynchronize();
                stopTimeAndPrint(&timer, "");



        }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



        // Copy data from GPU
        startTime(&timer);

        // TODO
        inBuffer->numRows = tmpInBuffer.numRows ;
        inBuffer->numCols = tmpInBuffer.numCols ;
        inBuffer->nnz = tmpInBuffer.nnz;
        inBuffer->capacity  = tmpInBuffer.capacity ;
        cudaMemcpy(inBuffer->rowPtrs, tmpInBuffer.rowPtrs, (tmpInBuffer.numRows+1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(inBuffer->colIdxs, tmpInBuffer.colIdxs, tmpInBuffer.numCols * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(inBuffer->values, tmpInBuffer.values, tmpInBuffer.numCols * sizeof(float), cudaMemcpyDeviceToHost);


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
        cudaFree(tmpInBuffer.rowPtrs);
        cudaFree(tmpInBuffer.colIdxs);
        cudaFree(tmpInBuffer.values);
        cudaFree(outBuffer_d);


        // cudaFree(tmpOutBuffer.rowIdxs);
        // cudaFree(tmpOutBuffer.colIdxs);
        // cudaFree(tmpOutBuffer.values);

        cudaFree(inBuffer_d);

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
        freeCSR(Y0);
        for (unsigned int layer = 0; layer < numLayers; ++layer) {
                freeCSC(W[layer]);
        }

        stopTimeAndPrint(&timer, "Deallocate memory");


}
