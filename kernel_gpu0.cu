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

// COOMatrix* sortCOO(COOMatrix *A){

//         // sorting rows
//          for (unsigned int i = 0; i < A->nnz; i++)
//                 for (unsigned int j = 0; j < A->nnz-i-1; j++)
//                 {    if (A->rowIdxs[j] > A->rowIdxs[j+1]){
//                                 unsigned int r = A->rowIdxs[j];
//                                 unsigned int c =  A->colIdxs[j];
//                                 float v = A->values[j];
//                                 A->rowIdxs[j] = A->rowIdxs[j+1];
//                                 A->colIdxs[j] = A->colIdxs[j+1];
//                                 A->values[j] = A->values[j+1];
//                                 A->rowIdxs[j+1] = r;
//                                 A->colIdxs[j+1] = c;
//                                 A->values[j+1] = v;
//                         }
//                 }

//          // sorting the col
//         // int count = 0;
//          int begin = 0;
//          for(unsigned int i  = 0 ;  i < A->nnz -1 ; i++)
//          {
//                  //count++;
//                  if(A->rowIdxs[i] != A->rowIdxs[i+1])
//                  {
//                          //sort the col
//                         for(int k = begin ;  k< i + begin; k++)
//                                 for (int m = begin ; m < i + begin - k -1 ;m++)
//                                         if(A->colIdxs[m] > A->colIdxs[m+1]){
//                                                 unsigned int c = A->colIdxs[m];
//                                                 float v = A->values[m];
//                                                 A->colIdxs[m] = A->colIdxs[m+1];
//                                                 A->values[m] = A->values[m+1];
//                                                 A->colIdxs[m+1] =c;
//                                                 A->values[m+1] = v;

//                                         }

//                         // count = 0;
//                         begin= i+1;
//                 }


//         }
//         return A;



//  }
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

        // //outBuffer_d allocation
        // COOMatrix* outBuffer_d;
        // COOMatrix tmpOutBuffer;
        // //cudaMalloc((void**)&outBuffer_d,sizeof(COOMatrix));
        // tmpOutBuffer.numRows = outBuffer->numRows;
        // tmpOutBuffer.numCols = outBuffer->numCols;
        // tmpOutBuffer.nnz = outBuffer->nnz;
        // tmpOutBuffer.capacity = outBuffer->capacity;
        // cudaMalloc((void**)&tmpOutBuffer.rowIdxs, (outBuffer->capacity) * sizeof(unsigned int));
        // cudaMalloc((void**)&tmpOutBuffer.colIdxs, (outBuffer->capacity) * sizeof(unsigned int));
        // cudaMalloc((void**)&tmpOutBuffer.values, (outBuffer->capacity) * sizeof(float));
        

        // cudaMemcpy(&(tmpOutBuffer.numRows),&(outBuffer->numRows),sizeof(unsigned int),cudaMemcpyHostToDevice);
        // cudaMemcpy(&(tmpOutBuffer.numCols),&(outBuffer->numCols),sizeof(unsigned int),cudaMemcpyHostToDevice);
        // cudaMemcpy(&(tmpOutBuffer.nnz),&(outBuffer->nnz),sizeof(unsigned int),cudaMemcpyHostToDevice);

        // cudaMemcpy(tmpOutBuffer.rowIdxs, outBuffer->rowIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
        // cudaMemcpy(tmpOutBuffer.colIdxs, outBuffer->colIdxs, outBuffer->capacity         * sizeof(unsigned int), cudaMemcpyHostToDevice);
        // cudaMemcpy(tmpOutBuffer.values, outBuffer->values, outBuffer->capacity * sizeof(float), cudaMemcpyHostToDevice);

        

        // cudaMalloc(&outBuffer_d, sizeof(COOMatrix*));

        // cudaMemcpy(outBuffer_d,&tmpOutBuffer,sizeof(COOMatrix),cudaMemcpyHostToDevice);

        //cudaMemcpy(&(outBuffer_d->nnz), &(tmpOutBuffer.nnz), sizeof(unsigned int*), cudaMemcpyHostToDevice);

        //cudaMemcpy(&out_nnz_d, &out_nnz_h, sizeof(unsigned int*), cudaMemcpyHostToDevice);
        ////////////////////////////
        // COOMatrix *outBuffer_d;
	// unsigned int* out_rowIdxs_d;
	// unsigned int* out_colIdxs_d;
        // float* out_values_d;
        // // outBuffer_d->numRows = outBuffer->numRows;
        // // outBuffer_d->numCols = outBuffer->numCols;
        // // outBuffer_d->nnz = outBuffer->nnz;
        // // outBuffer_d->capacity = outBuffer->capacity;
	// cudaMalloc((void**)&outBuffer_d, sizeof(COOMatrix));
	// cudaMalloc((void**)&out_rowIdxs_d, outBuffer->capacity * sizeof(unsigned int));
	// cudaMalloc((void**)&out_colIdxs_d, outBuffer->capacity * sizeof(unsigned int));
	// cudaMalloc((void**)&out_values_d, outBuffer->capacity * sizeof(float));

	// //copying outbuffer
	// cudaMemcpy(outBuffer_d, outBuffer, sizeof(COOMatrix), cudaMemcpyHostToDevice);
	// cudaMemcpy(out_rowIdxs_d, outBuffer->rowIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
	// cudaMemcpy(out_colIdxs_d, outBuffer->colIdxs, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
	// cudaMemcpy(out_values_d, outBuffer->values, outBuffer->capacity * sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(&(outBuffer_d->rowIdxs), &out_rowIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	// cudaMemcpy(&(outBuffer_d->colIdxs), &out_colIdxs_d, sizeof(unsigned int*), cudaMemcpyHostToDevice);
	// cudaMemcpy(&(outBuffer_d->values), &out_values_d, sizeof(float*), cudaMemcpyHostToDevice);
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
	cudaMemcpy(&out_nnz_d, &out_nnz_h, sizeof(unsigned int*), cudaMemcpyHostToDevice);
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

                spmspm <<<numBlocks, numThreadsPerBlock>>> (outBuffer_d, tmpInBuffer, W_d[layer], bias,out_nnz_d);


                cudaDeviceSynchronize();
                //cudaMemcpy(&tmpOutBuffer,outBuffer_d,sizeof(COOMatrix),cudaMemcpyDeviceToHost);
                
                
                // cudaMemcpy(&(tmpOutBuffer.rowIdxs),outBuffer_d->rowIdxs,sizeof(COOMatrix),cudaMemcpyDeviceToHost);
                // cudaMemcpy(&(tmpOutBuffer.colIdxs),outBuffer_d->colIdxs,sizeof(COOMatrix),cudaMemcpyDeviceToHost);
                // cudaMemcpy(&(tmpOutBuffer.values),outBuffer_d->values,sizeof(COOMatrix),cudaMemcpyDeviceToHost);
                //cudaMemcpy( &(tmpOutBuffer.nnz),&(outBuffer_d->nnz), sizeof(unsigned int*),cudaMemcpyDeviceToHost);
                
                // outBuffer->numRows =tmpOutBuffer.numRows;
                // outBuffer->numCols = tmpOutBuffer.numCols ;
                // outBuffer->nnz = tmpOutBuffer.nnz ;
                // outBuffer->capacity = tmpOutBuffer.capacity ;
                // cudaMemcpy(outBuffer->rowIdxs, tmpOutBuffer.rowIdxs,outBuffer->capacity * sizeof(unsigned int),cudaMemcpyDeviceToHost);
                // cudaMemcpy(outBuffer->colIdxs, tmpOutBuffer.colIdxs,outBuffer->capacity * sizeof(unsigned int),cudaMemcpyDeviceToHost);
                // cudaMemcpy(outBuffer->values, tmpOutBuffer.values,outBuffer->capacity * sizeof(unsigned int),cudaMemcpyDeviceToHost);
                // cudaMemcpy(outBuffer->rowIdxs, out_rowIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                // cudaMemcpy(outBuffer->colIdxs, out_colIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                // cudaMemcpy(outBuffer->values, out_values_d, outBuffer->capacity * sizeof(float), cudaMemcpyDeviceToHost);


                cudaMemcpy(outBuffer->rowIdxs, out_rowIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                cudaMemcpy(outBuffer->colIdxs, out_colIdxs_d, outBuffer->capacity * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                cudaMemcpy(outBuffer->values, out_values_d, outBuffer->capacity * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(out_nnz_h, out_nnz_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                outBuffer->nnz = *out_nnz_h;
                printf("nnz %d\n", outBuffer->nnz);
                for(int i =0; i<outBuffer->nnz;++i){
                        printf(" i = %d, row = %d, col = %d\n", i,outBuffer->rowIdxs[i],outBuffer->colIdxs[i]);
                }

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
