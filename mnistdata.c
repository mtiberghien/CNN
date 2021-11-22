#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "include/mnistdata.h"
#include "include/tensor.h"

dataset* getMNISTData(int limit, short test)
{
    FILE * fp;
    char * line = NULL;
    char* filename = test ? "../../datasets/MNIST/mnist_test.csv":"../../datasets/MNIST/mnist_train.csv";
    size_t len = 0;
    ssize_t read;
    tensor * features = (tensor*)malloc(sizeof(tensor));
    char** labels = (char**)malloc(sizeof(char*));
    fp = fopen(filename, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    int ln = 0;
    //skip headers
    getline(&line, &len, fp); 
    int size = 784;
    while ((read = getline(&line, &len, fp)) != -1) {
        if(read>1){
            features = (tensor*) realloc(features, (ln+1) * sizeof(tensor));
            labels = (char**) realloc(labels, (ln+1) * sizeof(char*));
            initialize_tensor(&features[ln], size);
            char delim[]=",";
            int column = 0;
            int i=0;
            char *ptr = strtok(line, delim);
            while(ptr != NULL && i < size)
            {
                if(column > 0){
                    features[ln].v[i] = strtod(ptr, NULL);
                    i++;
                }
                else
                {
                    labels[ln]= malloc(sizeof(char)*strlen(ptr));
                    strcpy(labels[ln], ptr);
                }
                ptr = strtok(NULL, delim);
                column++;
            }
            ln++;
            if(limit==ln)
            {
                break;
            }
        }
    }

    fclose(fp);
    if (line)
        free(line);
    dataset* result = (dataset*)malloc(sizeof(dataset));
    result->features = features;
    result->labels = labels;
    result->labels_categorical = to_categorical(labels, ln);
    result->n_entries = ln;
    return result;
}