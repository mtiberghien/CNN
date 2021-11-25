#ifndef CNN_TENSOR
#define CNN_TENSOR

#include <stdio.h>

//The data structure used in a neural network
typedef struct tensor{
    double* v;
    int size;
} tensor;

//Clear memory of a tensor
void clear_tensor(tensor*);
//Clear memory of a collection of tensors
void clear_tensors(tensor*,int);

double sum(tensor*, double(*func)(double));
double max(tensor*);
tensor* sub(tensor* tensor, double value);
void initialize_tensor(tensor* tensor, int size);
void print_tensor(tensor*);
tensor* to_categorical(char** labels, int n_labels);
int arg_max(tensor* tensor);
void apply_func(tensor* tensor, double(*func)(double));
void mult_tensor_func(tensor* tensor_dest, tensor* tensor_source, double(*func)(double));
void save_tensor(FILE* fp, tensor* tensor);
void read_tensor(FILE* fp, tensor* tensor, int size);
#endif

