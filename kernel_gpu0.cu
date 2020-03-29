// row+1; swapping; nnzidx; syncthreads
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"


#define THRESHOLD 0.000001
#define YMAX 32
#define threads 512


__global__ void spmspm(CSRMatrix *result, CSRMatrix *A, CSCMatrix *B, float bias, unsigned int *nnzIdx) {
    unsigned int r= blockIdx.y*blockDim.y +threadIdx.y;
    unsigned int c= blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int rowPtrA;
	unsigned int nnzA;

	if(r < A->numRows){

		rowPtrA = A->rowPtrs[r];

		nnzA = A->rowPtrs[r + 1] - rowPtrA;
			


		if(nnzA>0) { // if a row is not all zeros , we do computation otherwise we skip row
			printf("nnzA %d\n", nnzA);

			//ptrs to cols and vals of A[r]
			unsigned int* colIdxsA = A->colIdxs + rowPtrA;
			float* valueA = A->values + rowPtrA;


			//we will take one column of B

			unsigned int colPtrB = B->colPtrs[c];
			unsigned int nnzB = B->colPtrs[c + 1] - colPtrB;


			if( c < B->numCols) {
				if(nnzB>0) { // if a col in B is not all zeros, we do computation otherwise skip
					//ptrs to rows and vals of B[c]
					printf("nnzB %d\n", nnzB);
                    unsigned int* rowIdxsB = B->rowIdxs + colPtrB;
                    float* valueB = B->values + colPtrB;
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
                            // if(*nnzIdx >= result->capacity) { // if you fill the whole capacity for the result
                            //     expandCSRCapacity(result, 2*result->capacity);//expand result by double it's original capacity
                            // }
                            result->colIdxs[*nnzIdx] = c;
                            result->values[*nnzIdx] = sum;
                            atomicAdd(nnzIdx,1); //counts how many non zero elements I have 
                        }    
                    }
				}
				result->rowPtrs[r + 1] = *nnzIdx;//takes care of row ptr for result ()
				__syncthreads();
			}
		}
		result->nnz = *nnzIdx;
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
	CSRMatrix *tmp = createEmptyCSR(Y0->numRows, Y0->numCols, 2 * Y0->nnz);
	CSRMatrix *inBuffer = Y0;
	CSRMatrix *outBuffer = tmp;
	stopTimeAndPrint(&timer, "Allocate temporary buffer");



	// Allocate GPU memory
	startTime(&timer);

	//result_d allocation
	//Vector *result_d;
	//result_d->nnz = result->nnz;
	//result_d->capacity = result->capacity;
	//cudaMalloc((void**)&result_d->data, result->capacity * sizeof(unsigned int));

	//inBuffer_d allocation
	CSRMatrix inBuffer_d;
	inBuffer_d.numRows = inBuffer->numRows;
	inBuffer_d.numCols = inBuffer->numCols;
	inBuffer_d.nnz = inBuffer->nnz;
	inBuffer_d.capacity = inBuffer->capacity;
	cudaMalloc((void**)&inBuffer_d.rowPtrs, (inBuffer_d.numRows + 1) * sizeof(unsigned int));
	cudaMalloc((void**)&inBuffer_d.colIdxs, inBuffer_d.numCols * sizeof(unsigned int));
	cudaMalloc((void**)&inBuffer_d.values, inBuffer_d.numCols * sizeof(float));

	//outBuffer_d allocation
	CSRMatrix outBuffer_d;
	outBuffer_d.numRows = outBuffer->numRows;
	outBuffer_d.numCols = outBuffer->numCols;
	outBuffer_d.nnz = outBuffer->nnz;
	outBuffer_d.capacity = outBuffer->capacity;
	cudaMalloc((void**)&outBuffer_d.rowPtrs, (outBuffer_d.numRows + 1) * sizeof(unsigned int));
	cudaMalloc((void**)&outBuffer_d.colIdxs, outBuffer_d.numCols * sizeof(unsigned int));
	cudaMalloc((void**)&outBuffer_d.values, outBuffer_d.numCols * sizeof(float));

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

	cudaDeviceSynchronize();
	stopTime(&timer);
	printElapsedTime(timer, "Allocation time on GPU Memory");

	// Copy data to GPU
	startTime(&timer);
	
	//for result
	//cudaMemcpy(result_d->data, result->data, result_d->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
	

	//for inbuffer
	cudaMemcpy(inBuffer_d.rowPtrs, inBuffer->rowPtrs, inBuffer_d.numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(inBuffer_d.colIdxs, inBuffer->colIdxs, inBuffer_d.numCols * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(inBuffer_d.values, inBuffer->values, inBuffer_d.numCols * sizeof(float), cudaMemcpyHostToDevice);

	//for outbuffer
	cudaMemcpy(outBuffer_d.rowPtrs, outBuffer->rowPtrs, outBuffer_d.numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(outBuffer_d.colIdxs, outBuffer->colIdxs, outBuffer_d.numCols * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(outBuffer_d.values, outBuffer->values, outBuffer_d.numCols * sizeof(float), cudaMemcpyHostToDevice);

	//for Weights
	for (unsigned int layer = 0; layer < numLayers; ++layer) {
		cudaMemcpy(W_d[layer].colPtrs, W[layer]->colPtrs, W_d[layer].numCols * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(W_d[layer].rowIdxs, W[layer]->rowIdxs, W_d[layer].numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(W_d[layer].values, W[layer]->values, W_d[layer].numRows * sizeof(float), cudaMemcpyHostToDevice);
	}

	cudaDeviceSynchronize();
	stopTime(&timer);
	printElapsedTime(timer, "Copy to GPU time");

	//kernel loop

	// Loop over layers
	for (unsigned int layer = 0; layer < numLayers; ++layer) {

		// SpMSpM
		printf("Computing layer %u (SpMSpM)", layer);
		startTime(&timer);
		unsigned int nnzIdx=0;
		
		//do kernel call instead
		//int outputSize = inBuffer_d->numRows * W_d[layer]->numCols;

		dim3 numThreadsPerBlock(threads, threads);
		dim3 numBlocks((W_d[layer].numCols + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,(inBuffer_d.numRows + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
		//int numBlocks = (outputSize + numThreadsPerBlock - 1)/numThreadsPerBlock ;
		spmspm <<<numBlocks, numThreadsPerBlock>>> (&outBuffer_d,&inBuffer_d,&W_d[layer],bias,&nnzIdx);
		
		cudaDeviceSynchronize();
		stopTimeAndPrint(&timer, "");

		// Swap buffers
		CSRMatrix t = inBuffer_d;
		inBuffer_d = outBuffer_d;
		outBuffer_d = t;

	}

	
	// Copy data from GPU
	startTime(&timer);

	// TODO
	
	cudaMemcpy(inBuffer->rowPtrs, inBuffer_d.rowPtrs, inBuffer_d.numRows * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(inBuffer->colIdxs, inBuffer_d.colIdxs, inBuffer_d.numCols * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(inBuffer->values, inBuffer_d.values, inBuffer_d.numCols * sizeof(float), cudaMemcpyDeviceToHost);


	cudaDeviceSynchronize();
	stopTime(&timer);
	printElapsedTime(timer, "Copy from GPU time");

	//CPU 
	// Find nonzero rows
	startTime(&timer);
	findNonzeroRows(result, inBuffer);
	stopTimeAndPrint(&timer, "Find nonzero rows");

	// Free GPU memory
	startTime(&timer);

	cudaFree(inBuffer_d.rowPtrs);
	cudaFree(inBuffer_d.colIdxs);
	cudaFree(inBuffer_d.values);
	cudaFree(outBuffer_d.rowPtrs);
	cudaFree(outBuffer_d.colIdxs);
	cudaFree(outBuffer_d.values);
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
	freeCSR(tmp);
	stopTimeAndPrint(&timer, "Deallocate memory");
	

}

