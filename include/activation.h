#ifndef ACTIVATION_CNN
#define ACTIVATION_CNN

#include "tensor.h"
#include <stdlib.h>

typedef enum activation_type {RELU,SOFTMAX} activation_type;

//Represent a layer activation function and its derivative
typedef struct activation{
    activation_type type;
    //Specific activation calculation
    tensor* (*activation_func)(tensor* input);
    //Specific activation derivative calculation
    tensor* (*activation_func_prime)(tensor* activation_input);
    tensor* (*activation_backward_propagation)(tensor* activation_input, tensor* gradient, tensor* output, struct activation* activation);
} activation;

tensor* activation_backward_propagation(tensor* activation_input, tensor* gradient,tensor* output, activation* activation);
tensor* activation_func_relu(tensor* input);
tensor* activation_func_prime_relu(tensor* activation_input);
activation* build_activation_relu();

tensor* activation_func_softmax(tensor* input);
tensor* backward_propagation_softmax(tensor* activation_input, tensor* gradient,tensor* output, activation* activation);
activation* build_activation_softmax();

#endif