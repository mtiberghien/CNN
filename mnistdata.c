#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "include/mnistdata.h"
#include "include/tensor.h"

dataset* getMNISTData(int limit, short test)
{
    FILE * fp;
    char * line = NULL;
    char path[100]= "../../datasets/MNIST/";
    char* filename = test ? "mnist_test.csv":"mnist_train.csv";
    char* filepath = strcat(path, filename);
    printf("reading %s...\n", filename);
    size_t len = 0;
    ssize_t read;
    tensor * features = (tensor*)malloc(sizeof(tensor));
    char** labels = (char**)malloc(sizeof(char*));
    dataset* result = (dataset*)malloc(sizeof(dataset));
    result->n_features = 784;
    fp = fopen(filepath, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    int ln = 0;
    double norm_factor = ((double)1)/255;
    //skip headers
    getline(&line, &len, fp); 
    while ((read = getline(&line, &len, fp)) != -1) {
        if(read>1){
            features = (tensor*) realloc(features, (ln+1) * sizeof(tensor));
            labels = (char**) realloc(labels, (ln+1) * sizeof(char*));
            initialize_tensor(&features[ln], result->n_features);
            char delim[]=",";
            int column = 0;
            int i=0;
            char *ptr = strtok(line, delim);
            while(ptr != NULL && i < result->n_features)
            {
                if(column > 0){
                    features[ln].v[i] = strtod(ptr, NULL) * norm_factor;
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
    printf("read %d lines from %s.\n", ln, filename);
    result->features = features;
    result->labels = labels;
    result->labels_categorical = to_categorical(labels, ln);
    result->n_entries = ln;
    return result;
}