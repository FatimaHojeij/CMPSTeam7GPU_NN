#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"


#define THRESHOLD 0.000001
#define YMAX 32
#define threads 512


__global__ void spmspm(COOMatrix *result, CSRMatrix A, CSCMatrix B, float bias) {
    unsigned int r= blockIdx.y*blockDim.y +threadIdx.y;
    unsigned int c= blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int rowPtrA;
    unsigned int nnzA;
        if(r < A.numRows && c < B.numCols){
                rowPtrA = A.rowPtrs[r];
                nnzA = A.rowPtrs[r + 1] - rowPtrA;
                if(nnzA>0) { // if a row is not all zeros , we do computation otherwise we skip row
                        //ptrs to cols and vals of A[r]
                        unsigned int* colIdxsA = A.colIdxs + rowPtrA;
                        float* valueA = A.values + rowPtrA;
                        //we will take one column of B
                        unsigned int colPtrB = B.colPtrs[c];
                        unsigned int nnzB = B.colPtrs[c + 1] - colPtrB;
                        if(nnzB>0) { // if a col in B is not all zeros, we do computation otherwise skip//ptrs to rows and vals of B[c]
                                unsigned int* rowIdxsB = B.rowIdxs + colPtrB;
                                float* valueB = B.values + colPtrB;
                                // Loop and find intersection
                                float sum = 0.0f;
                                unsigned int ia = 0, ib = 0;
                                while(ia < nnzA && ib < nnzB) { // loops over all non zeros from A and B and stop when there is no more non zero
                                        unsigned int colIdx = colIdxsA[ia]; //single item col index from A
                                        unsigned int rowIdx = rowIdxsB[ib]; //single item row index from B
                                        if(colIdx < rowIdx) {
                                                ia++;
                                        } else if(colIdx > rowIdx) {
                                                ib++;
                                        } else {
                                                sum += valueA[ia]*valueB[ib];// do the multiplication of the row that matches the column
                                                ia++;
                                                ib++;
                                        }
                                }
                                if(sum > THRESHOLD || sum < -THRESHOLD) { //if not smaller than abs(threshold)
                                        sum += bias; //add to it the bias
                                        //Remove negative and zero values
                                        if(sum > 0) {//if end result is positive otherwise I also do not want to add it to result
                                                if(sum>YMAX) { //make sure it is on an upper limit
                                                        sum = YMAX;
                                                }
                                                unsigned int nnzIndxTemp = atomicAdd(&(result->nnz),1); //counts how many non zero elements I have
                                                result->rowIdxs[nnzIndxTemp] = r;
                                                result->colIdxs[nnzIndxTemp] = c;
                                                result->values[nnzIndxTemp] = sum;
                                        }
                                }
                        }

                }
        }
}

COOMatrix* sortCOO(COOMatrix *A){

        // sorting rows
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

         // sorting the col
        // int count = 0;
         int begin = 0;
         for(unsigned int i  = 0 ;  i < A->nnz -1 ; i++)
         {
                 //count++;
                 if(A->rowIdxs[i] != A->rowIdxs[i+1])
                 {
                         //sort the col
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

                        // count = 0;
                        begin= i+1;
                }


        }
        return A;



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
	
	
	//inBuffer_d allocation
	CSRMatrix* inBuffer_d;
    unsigned int* in_rowPtrs_d;
    unsigned int* in_colIdxs_d;
    float* in_values_d;
	cudaMalloc((void**) &inBuffer_d, sizeof(CSRMatrix));
    cudaMalloc((void**) &in_rowPtrs_d, (inBuffer->numRows + 1) * sizeof(unsigned int));
    cudaMalloc((void**) &in_colIdxs_d, inBuffer->numCols * sizeof(unsigned int));
    cudaMalloc((void**) &in_values_d, inBuffer->numCols * sizeof(float));
	
	
	//outBuffer_d allocation
    COOMatrix *outBuffer_d;
	unsigned int* out_rowIdxs_d;
    unsigned int* out_colIdxs_d;
    float* out_values_d;
    cudaMalloc((void**)&outBuffer_d, sizeof(COOMatrix));
    cudaMalloc((void**)&out_rowIdxs_d, outBuffer->capacity * sizeof(unsigned int));
    cudaMalloc((void**)&out_colIdxs_d, outBuffer->capacity * sizeof(unsigned int));
    cudaMalloc((void**)&out_values_d, outBuffer->capacity * sizeof(float));
		
	
	
	//copying inbuffer
	cudaMemcpy(inBuffer_d, inBuffer, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
	cudaMemcpy(in_rowPtrs_d, inBuffer->rowPtrs, (inBuffer->numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(in_colIdxs_d, inBuffer->colIdxs, inBuffer->numCols * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(in_values_d, inBuffer->values, inBuffer->numCols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&(inBuffer_d->rowPtrs), &in_rowPtrs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(inBuffer_d->colIdxs), &in_colIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(inBuffer_d->values), &in_values_d, sizeof(float*), cudaMemcpyHostToDevice);
	printElapsedTime(timer, "For inBuffer");
	
	//copying outbuffer
    cudaMemcpy(outBuffer_d, outBuffer, sizeof(COOMatrix), cudaMemcpyHostToDevice);
	cudaMemcpy(out_rowIdxs_d, outBuffer->rowIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_colIdxs_d, outBuffer->colIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_values_d, outBuffer->values, outBuffer->capacity * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&(outBuffer_d->rowIdxs), &out_rowIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(outBuffer_d->colIdxs), &out_colIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(outBuffer_d->values), &out_values_d, sizeof(float*), cudaMemcpyHostToDevice);
    printElapsedTime(timer, "For outBuffer");
	
	cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation & Copy to GPU time");
	
	
	for (unsigned int layer = 0; layer < numLayers; ++layer) {
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
		
		
		dim3 numThreadsPerBlock(threads, threads);
        dim3 numBlocks((W[layer]->numCols + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,(inBuffer_d.numRows + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
        spmspm <<<numBlocks, numThreadsPerBlock>>> (outBuffer_d, *inBuffer_d, *W_d, bias);
        cudaDeviceSynchronize();
        stopTimeAndPrint(&timer, "");
		
		stopTimeAndPrint(&timer, "For Out Buffer");
		cudaMemcpy(outBuffer, outBuffer_d, sizeof(COOMatrix), cudaMemcpyDeviceToHost);
		//struct fields as variables(?)
		cudaMemcpy(outBuffer->rowIdxs, out_rowIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(outBuffer->colIdxs, out_colIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(outBuffer->values, out_values_d, outBuffer->capacity * sizeof(float), cudaMemcpyDeviceToHost);
		
		
		
		stopTimeAndPrint(&timer, "For Sort");
       		inBuffer = createCSRfromCOO(sortCOO(outBuffer));
       		stopTimeAndPrint(&timer, "Out of sort");
		
		//do we need to malloc again (?)
		cudaMemcpy(inBuffer_d, inBuffer, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
		cudaMemcpy(in_rowPtrs_d, inBuffer->rowPtrs, (inBuffer->numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(in_colIdxs_d, inBuffer->colIdxs, inBuffer->numCols * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(in_values_d, inBuffer->values, inBuffer->numCols * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(&(inBuffer_d->rowPtrs), &in_rowPtrs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
		cudaMemcpy(&(inBuffer_d->colIdxs), &in_colIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
		cudaMemcpy(&(inBuffer_d->values), &in_values_d, sizeof(float*), cudaMemcpyHostToDevice);
		
		
		outBuffer = createEmptyCOO(inBuffer->numRows, inBuffer->numCols, 2*inBuffer->capacity);
		
		
		//do we need to malloc again (?)
		cudaMemcpy(outBuffer_d, outBuffer, sizeof(COOMatrix), cudaMemcpyHostToDevice);
		cudaMemcpy(out_rowIdxs_d, outBuffer->rowIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(out_colIdxs_d, outBuffer->colIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(out_values_d, outBuffer->values, outBuffer->capacity * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(&(outBuffer_d->rowIdxs), &out_rowIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
		cudaMemcpy(&(outBuffer_d->colIdxs), &out_colIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
		cudaMemcpy(&(outBuffer_d->values), &out_values_d, sizeof(float*), cudaMemcpyHostToDevice);
		
		cudaFree(w_colPtrs_d);
		cudaFree(w_rowIdxs_d);
		cudaFree(w_values_d);
		cudaFree(W_d);
	}
	
	// Copy data from GPU
    startTime(&timer);
	
	cudaMemcpy(inBuffer, inBuffer_d, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
	//struct fields as variables(?)
	cudaMemcpy(inBuffer->rowPtrs, in_rowPtrs_d, (inBuffer->numRows + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(inBuffer->colIdxs, in_colIdxs_d, inBuffer->numCols * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(inBuffer->values, in_values_d, inBuffer->numCols * sizeof(float), cudaMemcpyDeviceToHost);
	//copy pointers back (??)
	cudaMemcpy(&in_rowPtrs_d, &(inBuffer_d->rowPtrs), sizeof(unsigned int*), cudaMemcpyDeviceToHost);
	cudaMemcpy(&in_colIdxs_d, &(inBuffer_d->colIdxs), sizeof(unsigned int*), cudaMemcpyDeviceToHost);
	cudaMemcpy(&in_values_d, &(inBuffer_d->values), sizeof(float*), cudaMemcpyDeviceToHost);
	
	
	
	cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");
	
	
	// Find nonzero rows
    startTime(&timer);
    findNonzeroRows(result, inBuffer);
	stopTimeAndPrint(&timer, "Find nonzero rows");

    // Free GPU memory
        startTime(&timer);

        cudaFree(in_rowPtrs_d);
        cudaFree(in_colIdxs_d);
        cudaFree(in_values_d);
		cudaFree(inBuffer_d);
        cudaFree(out_rowIdxs_d);
        cudaFree(out_colIdxs_d);
        cudaFree(out_values_d);
        cudaFree(outBuffer);
        
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
