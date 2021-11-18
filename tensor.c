#include "include/tensor.h"
#include <stdlib.h>
#include "float.h"

//Free memory of a tensor
void clear_tensor(tensor* tensor){
    free(tensor->v);
}

//Free memory for a batch of tensors
void clear_tensors(tensor* tensors, int n_tensor){
    for(int i=0;i<n_tensor;i++){
        clear_tensor(&tensors[i]);
    }
    free(tensors);
}

//Sum the result of a function on a tensor
double sum(tensor* tensor, double(*func)(double x))
{
    double result = 0;
    for(int i=0;i<tensor->size;i++)
    {
        result+=func(tensor->v[i]);
    }
    return result;
}

double max(tensor* tensor)
{
    double result = -DBL_MAX;
    for(int i=0;i<tensor->size;i++)
    {
        result = tensor->v[i]>result ? tensor->v[i]:result;
    }
    return result;
}

tensor* sub(tensor* tensor, double value)
{
    for(int i=0;i<tensor->size;i++)
    {
        tensor->v[i]-=value;
    }
    return tensor;
}