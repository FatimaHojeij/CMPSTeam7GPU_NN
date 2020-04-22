
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"

#define THRESHOLD 0.000001
#define YMAX 32

void spmspm(CSRMatrix *result, CSRMatrix *A, CSCMatrix *B, float bias) {
	unsigned int nnzIdx = 0;
    for(unsigned int r = 0; r < A->numRows; r++) { //loops over the rows of A
        unsigned int rowPtrA = A->rowPtrs[r];
        unsigned int nnzA = A->rowPtrs[r + 1] - rowPtrA;
        if(nnzA>0) { // if a row is not all zeros , we do computation otherwise we skip row
            unsigned int* colIdxsA = A->colIdxs + rowPtrA;
            float* valueA = A->values + rowPtrA;
            for(unsigned int c = 0; c < B->numCols; c++) { // loops over the columns of B
                unsigned int colPtrB = B->colPtrs[c];
                unsigned int nnzB = B->colPtrs[c + 1] - colPtrB;
                if(nnzB>0) { // if a col in B is not all zeros, we do computation otherwise skip
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

					//RELU SECTION HERE
					
					//do not add to result everything that is smaller than 0 (if u put in RELU everything smaller than 0 will become 0)
                    if(sum > THRESHOLD || sum < -THRESHOLD) { //if not smaller than abs(threshold)
                        sum += bias; //add to it the bias
                        //Remove negative and zero values
                        if(sum > 0) {//if end result is positive otherwise I also do not want to add it to result
							if(sum>YMAX) { //make sure it is on an upper limit
                                sum = YMAX;
                            }
                            if(nnzIdx >= result->capacity) { // if you fill the whole capacity for the result
                                expandCSRCapacity(result, 2*result->capacity);//expand result by double it's original capacity
                            }
                            result->colIdxs[nnzIdx] = c;
                            result->values[nnzIdx] = sum;
                            ++nnzIdx; //counts how many non zero elements I have 
                        }    
                    }

                }
            }
        }

		/*
		rowptr:		0		3		5		(set in outer for loop, calculated in inner for loop)
		col:		2 3 5	2 4		1 4
		values:		1 2 3	1 2		4 5	

		*/

        result->rowPtrs[r + 1] = nnzIdx;//takes care of row ptr for result ()
    }
    result->nnz = nnzIdx;
}

//converts from CSRMatrix to Vector and a vector of indices where the row is not all zeros
void findNonzeroRows(Vector* v, CSRMatrix* A) {
    unsigned int nnz = 0;
    for(unsigned int r = 0; r < A->numRows; ++r) {
        unsigned int rowPtrA = A->rowPtrs[r];
        unsigned int nnzA = A->rowPtrs[r + 1] - rowPtrA;
        if(nnzA > 0) {
            if(nnz >= v->capacity) {
                expandVectorCapacity(v, 2*v->capacity);
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
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        W[layer] = createCSCfromCOO(layerWeights[layer]);
    }
    stopTimeAndPrint(&timer, "Convert weights to CSR");

    // Double buffers
    startTime(&timer);
    CSRMatrix *tmp = createEmptyCSR(Y0->numRows, Y0->numCols, 2*Y0->nnz);
    CSRMatrix *inBuffer  = Y0;
    CSRMatrix *outBuffer = tmp;
    stopTimeAndPrint(&timer, "Allocate temporary buffer");
        
    // Loop over layers
    for(unsigned int layer = 0; layer < numLayers; ++layer) {

        // SpMSpM
        printf("Computing layer %u (SpMSpM)", layer);
        startTime(&timer);
        spmspm(outBuffer, inBuffer, W[layer], bias);
        stopTimeAndPrint(&timer, "");

        // Swap buffers
        CSRMatrix *t = inBuffer;
        inBuffer = outBuffer;

        FILE* f = fopen("./binning_cpu.txt","w");

		for(int i =0; i<inBuffer->numRows;++i){

			fprintf(f,"%d\t%d:\n",i,inBuffer->rowPtrs[i]);
			int rowPtr = inBuffer->rowPtrs[i];
			int nnz = inBuffer->rowPtrs[i+1]-inBuffer->rowPtrs[i];

			for(int j = rowPtr;j<rowPtr+nnz;++j){

				fprintf(f,"%d\n",inBuffer->colIdxs[j]);
			}

		}

		fclose(f);

		return;


        outBuffer = t;



    }

    // Find nonzero rows
    startTime(&timer);
    findNonzeroRows(result, inBuffer);
    stopTimeAndPrint(&timer, "Find nonzero rows");

    // Free buffers
    startTime(&timer);
    freeCSR(Y0);
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        freeCSC(W[layer]);
    }
    freeCSR(tmp);
    stopTimeAndPrint(&timer, "Deallocate memory");

}

