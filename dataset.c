#include "include/dataset.h"
#include <stdlib.h>

void clear_dataset(dataset* dataset)
{
    clear_tensors(dataset->features, dataset->n_entries);
    clear_tensors(dataset->labels_categorical, dataset->n_entries);
    for(int i=0;i<dataset->n_entries;i++)
    {
        free(dataset->labels[i]);
    }
    free(dataset->labels);
    free(dataset->features);
    free(dataset->labels_categorical);
    free(dataset);
}