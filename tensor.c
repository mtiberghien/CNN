#include "include/tensor.h"
#include <stdlib.h>

//Free memory of a tensor
void clear_tensor(tensor tensor){
    free(tensor.v);
}

//Free memory for a batch of tensors
void clear_tensors(tensor* tensors, int n_tensor){
    for(int i=0;i<n_tensor;i++){
        clear_tensor(tensors[i]);
    }
    free(tensors);
}