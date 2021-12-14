#ifndef ACTIVATION_CNN
#define ACTIVATION_CNN

#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>

typedef enum activation_type {RELU, TANH, SIGMOID, SOFTMAX} activation_type;

//Represent a layer activation function and its derivative
typedef struct activation{
    activation_type type;
    //Specific activation calculation
    double (*activation_func)(double);
    double (*activation_func_prime)(double);
    //Specific activation derivative calculation
    tensor* (*activation_forward)(tensor* activation_input, struct activation* activation);
    tensor* (*activation_backward_propagation)(const tensor* activation_input, tensor* gradient, tensor* output, struct activation* activation);
} activation;

extern activation* build_activation(activation_type type);
void save_activation(FILE* fp, activation* activation);
activation* read_activation(FILE* fp);
tensor* activation_forward(tensor* input, activation* activation);
tensor* activation_backward_propagation(const tensor* activation_input, tensor* gradient, tensor* output, activation* activation);
activation* build_activation_relu();
activation* build_activation_sigmoid();
activation* build_activation_softmax();
activation* build_activation_tanh();
activation* build_default_activation(activation_type type);
#endif