#include "include/tensor.h"
#include <stdlib.h>
#include "float.h"
#include <stdio.h>
#include "include/common.h"

//Free memory of a tensor
void clear_tensor(tensor* tensor){
    free(tensor->v);
}

//Free memory for a batch of tensors
void clear_tensors(tensor* tensors, int n_tensor){
    for(int i=0;i<n_tensor;i++){
        clear_tensor(&tensors[i]);
    }
}

void apply_func(tensor* tensor, double(*func)(double))
{
    for(int i=0;i<tensor->size;i++)
    {
        tensor->v[i]=func(tensor->v[i]);
    }
}

void mult_tensor_func(tensor* tensor_dest, tensor* tensor_source, double(*func)(double))
{
    for(int i=0;i<tensor_dest->size;i++)
    {
        tensor_dest->v[i]*=func(tensor_source->v[i]);
    }
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

void initialize_tensor(tensor* tensor, int size)
{
    tensor->size = size;
    tensor->v=calloc(size,sizeof(double));
}

void print_tensor(tensor* tensor)
{
    for(int i=0;i<tensor->size;i++)
    {
        printf("%6.2f ",tensor->v[i]);
    }
    printf("\n");
}

tensor* to_categorical(char** labels, int n_labels)
{
    char** unique_labels = (char**)malloc(sizeof(char*));
    int count=0;
    for(int i=0;i<n_labels;i++)
    {
        if(index_of(unique_labels, count, labels[i])<0)
        {
            unique_labels = realloc(unique_labels, sizeof(char*)*++count);
            unique_labels[count-1]=labels[i];
        }
    }
    sort(unique_labels, count);
    tensor* result = malloc(sizeof(tensor)*n_labels);
    for(int i=0;i<n_labels;i++)
    {
        initialize_tensor(&result[i], count);
        result[i].v[index_of(unique_labels, count, labels[i])]=1;
    }
    free(unique_labels);
    return result;
}

int arg_max(tensor* tensor)
{
    double max = -DBL_MAX;
    int result = -1;
    for(int i=0;i<tensor->size;i++)
    {
        if(tensor->v[i]>max)
        {
            max = tensor->v[i];
            result = i;
        }
    }
    return result;
}


