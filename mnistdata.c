#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "include/mnistdata.h"
#include "include/tensor.h"
#include "include/progression.h"

dataset* getMNISTData(int limit, short test)
{
    FILE * fp;
    char * line = NULL;
    char path[100]= "../../datasets/MNIST/";
    char* filename = test ? "mnist_test.csv":"mnist_train.csv";
    char* filepath = strcat(path, filename);
    char header[100]="reading ";

    progression* progression = build_progression(limit, strcat(header, filename) );
    size_t len = 0;
    ssize_t read;
    tensor * features = (tensor*)malloc(sizeof(tensor));
    char** labels = (char**)malloc(sizeof(char*));
    dataset* result = (dataset*)malloc(sizeof(dataset));
    int n_features = 784;
    result->features_shape = build_shape(ThreeD);
    result->features_shape->sizes[0]=1;
    result->features_shape->sizes[1]=28;
    result->features_shape->sizes[2]=28;
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
            initialize_tensor(&features[ln], result->features_shape);
            char delim[]=",";
            int column = 0;
            int i=0;
            char *ptr = strtok(line, delim);
            tensor* feature = &features[ln];
            int* iterator = get_iterator(feature);
            while(ptr != NULL && i < n_features)
            {
                if(column > 0){
                    double d =  strtod(ptr, NULL) * norm_factor;
                    feature->set_value(feature, iterator,d);
                    iterator = feature->get_next(feature, iterator);
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
            free(iterator);
            ln++;
            progression->call_back(progression);
            if(limit==ln)
            {
                break;
            }
        }
    }
    fclose(fp);
    if (line)
        free(line);
    clear_progression(progression);
    printf("read %d lines from %s.\n", ln, filename);
    result->features = features;
    result->labels = labels;
    result->labels_categorical = to_categorical(labels, ln);
    result->n_entries = ln;
    return result;
}