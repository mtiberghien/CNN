#ifndef ACTIVATION_CNN
#define ACTIVATION_CNN

#include "tensor.h"
#include <stdlib.h>

//Represent a layer activation function and its derivative
typedef struct activation{
    //Specific activation calculation
    tensor* (*activation_func)(tensor* input);
    //Specific activation derivative calculation
    tensor* (*activation_func_prime)(tensor* output_error);
} activation;

tensor* activation_func_relu(tensor* input);
tensor* activation_func_prime_relu(tensor* output_error);
activation* build_activation_relu();

tensor* activation_func_softmax(tensor* input);
tensor* activation_func_prime_softmax(tensor* output_error);
activation* build_activation_softmax();

#endif