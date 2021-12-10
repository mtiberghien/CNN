#include "../../include/activation.h"

double relu(double x)
{
    return x<0?0:x;
}

double relu_prime(double x)
{
    return x<=0?0:1;
}

activation* build_activation_relu()
{
    activation* result = build_default_activation(RELU);
    result->activation_backward_propagation=activation_backward_propagation;
    result->activation_forward = activation_forward;
    result->activation_func=relu;
    result->activation_func_prime=relu_prime;
    return result;
}