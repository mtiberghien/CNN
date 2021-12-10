#ifndef DATASET_CNN
#define DATASET_CNN

#include "tensor.h"

typedef struct dataset{
    tensor * features;
    shape* features_shape;
    char** labels;
    tensor * labels_categorical;
    int n_entries;

} dataset;

void clear_dataset(dataset* dataset);
void free_dataset(dataset* dataset);
#endif