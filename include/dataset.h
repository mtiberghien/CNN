#ifndef DATASET_CNN
#define DATASET_CNN

#include "tensor.h"

typedef struct dataset{
    tensor * features;
    int n_features;
    char** labels;
    tensor * labels_categorical;
    int n_entries;

} dataset;

void clear_dataset(dataset* dataset);
#endif