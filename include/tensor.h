#ifndef CNN_TENSOR
#define CNN_TENSOR

#include <stdio.h>
#include "common.h"

typedef struct shape{
    int* sizes;
    dimension dimension;
} shape;

//The data structure used in a neural network
typedef struct tensor{
    double* v;
    shape* shape;
    double (*get_value)(const struct tensor*,int* iterator);
    void (*set_value)(const struct tensor*,int* iterator, double value);
    int* (*get_next)(const struct tensor*,int* iterator);
    short (*is_done)(const struct tensor*, int* iterator);
    void (*clear_tensor)(double* v, int* sizes);
} tensor;

//Clear memory of a tensor
void clear_tensor(tensor*);
//Clear memory of a collection of tensors
void clear_tensors(tensor*,int);
void clear_shape(shape*);
double sum(tensor*, double(*func)(double));
double max(tensor*);
tensor* sub(tensor* tensor, double value);
void initialize_tensor(tensor* tensor, shape* shape);
int* get_iterator(const tensor*);
void print_tensor(const tensor*);
tensor* to_categorical(char** labels, int n_labels);
int* arg_max(tensor* tensor);
void apply_func(tensor* tensor, double(*func)(double));
void mult_tensor_func(tensor* tensor_dest,const tensor* tensor_source, double(*func)(double));
void save_tensor(FILE* fp, tensor* tensor);
void read_tensor(FILE* fp, tensor* tensor);
shape* build_shape(dimension dim);
void save_shape(FILE* fp, shape* shape);
shape* read_shape(FILE* fp);
shape* clone_shape(const shape*);
short is_iterator_equal(int* i1, int*i2, dimension dimension);
void free_shape(shape*);
void free_tensor(tensor* tensor);
void free_tensors(tensor* tensors, int n_tensors);
#endif

