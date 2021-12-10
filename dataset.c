#include "include/dataset.h"
#include <stdlib.h>
#include "include/tensor.h"

//Clear data set memory
void clear_dataset(dataset* dataset)
{
    free_tensors(dataset->features, dataset->n_entries);
    free_tensors(dataset->labels_categorical, dataset->n_entries);
    for(int i=0;i<dataset->n_entries;i++)
    {
        free(dataset->labels[i]);
    }
    free(dataset->labels);
    free_shape(dataset->features_shape);
}

//Clear dataset memory. The dataset object is destroyed
void free_dataset(dataset* dataset)
{
    clear_dataset(dataset);
    free(dataset);
}