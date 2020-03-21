
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"
#include "verify.h"

int main(int argc, char** argv) {

    cudaDeviceSynchronize();

    // Parse arguments
	//default values if no argument provided
    const char*  inputDirectory = "data"; // get from directory dataset
    unsigned int numLayers = 120; // 120 layers
    unsigned int neuronsPerLayer = 1024; // 1024 neurons in a layer
    float  bias = -0.3; // bias term
    int opt;
    while((opt = getopt(argc, argv, "d:l:n:b:")) >= 0) { //command line if arguments provided
        switch(opt) {
            case 'd': inputDirectory  = optarg;       break;
            case 'l': numLayers       = atoi(optarg); break;
            case 'n': neuronsPerLayer = atoi(optarg); break;
            case 'b': bias            = atof(optarg); break;
            default:  fprintf(stderr, "\nUnrecognized option!\n");
                      exit(0);
        }
    }

    // File names
    unsigned int fileNameMaxSize = 100;
    char inputFileName[fileNameMaxSize];
    char categoriesFileName[fileNameMaxSize];
    char layerFileName[numLayers][fileNameMaxSize];
    snprintf(inputFileName, fileNameMaxSize, "%s/sparse-images-%d.tsv", inputDirectory, neuronsPerLayer);
    snprintf(categoriesFileName, fileNameMaxSize, "%s/neuron%d-l%d-categories.tsv", inputDirectory, neuronsPerLayer, numLayers);
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        snprintf(layerFileName[layer], fileNameMaxSize, "%s/neuron%d/n%d-l%d.tsv", inputDirectory, neuronsPerLayer, neuronsPerLayer, layer + 1);
    }

    // Read feature vectors  , reads X
    Timer timer;
    startTime(&timer);
    COOMatrix* featureVectors = createCOOFromFile(inputFileName, neuronsPerLayer);
    stopTimeAndPrint(&timer, "Reading feature vectors from file");

    // Read layer weight, reads W
    COOMatrix* layerWeights[numLayers]; // 120 layers -> 120 W's
    startTime(&timer);
    unsigned int numEdges =0;
    for(unsigned int layer = 0; layer < numLayers; ++layer) { //as much as we have layers , we have weight matrices
        layerWeights[layer] = createCOOFromFile(layerFileName[layer], neuronsPerLayer); // convert the W matrix as COO
        numEdges += layerWeights[layer]->nnz;
    }
    stopTimeAndPrint(&timer, "Reading layer weights from file");
    printf("Layers: %d, neurons/layer: %d, edges: %d\n", numLayers, neuronsPerLayer, numEdges);

    // Allocate output vector , final Y vector
    startTime(&timer);
    Vector* scores = createEmptyVector(featureVectors->numRows);
    stopTimeAndPrint(&timer, "Allocating output vector");

    // Perform computation
    startTime(&timer);
    sparseNN(scores, featureVectors, layerWeights, bias, numLayers);
    stopTimeAndPrintWithRate(&timer, "Total inference time", "edges", featureVectors->numRows*numEdges);

    // Store and verify result
    writeVectorToFile(scores, "scores.tsv");
    verify(scores, categoriesFileName);

    // Free data
    freeCOO(featureVectors);
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        freeCOO(layerWeights[layer]);
    }
    freeVector(scores);

    return 0;

}

