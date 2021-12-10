#include "../../include/activation.h"
#include <math.h>

double tanh_prime(double x)
{
    return 1-pow(tanh(x),(double)2);
}

activation* build_activation_tanh()
{
    activation* result = build_default_activation(TANH);
    result->activation_backward_propagation=activation_backward_propagation;
    result->activation_forward = activation_forward;
    result->activation_func=tanh;
    result->activation_func_prime=tanh_prime;
    return result;
}