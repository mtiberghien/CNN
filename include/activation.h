#ifndef ACTIVATION_CNN
#define ACTIVATION_CNN

#include "tensor.h"
#include <stdlib.h>

//Represent a layer activation function and its primitive
typedef struct activation{
    //Specific activation calculation
    double (*activation_func)(double);
    //Specific activation primitive calculation
    double (*activation_func_prime)(double);
} activation;

double activation_func_relu(double input);
double activation_func_prime_relu(double input);
activation* build_activation_relu();

#endif