#include "include/activation.h"
#include "include/tensor.h"

//Relu activation calculation
double activation_func_relu(double input)
{
    return input<0?0:input;
}

//Relu primitive calculation
double activation_func_prime_relu(double input)
{
    return input<=0?0:1;
}

activation* build_activation_relu(){
    activation* result = (activation*) malloc(sizeof(activation));
    result->activation_func=activation_func_relu;
    result->activation_func_prime=activation_func_prime_relu;
}