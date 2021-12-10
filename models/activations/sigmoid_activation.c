#include "../../include/activation.h"
#include <math.h>

//sigmoid calculation
double sigmoid(double x)
{
    return 1/(1+exp(-x));
}

//Apply this method to the sigmoid of a value will return the derivative of the sigmoid function
double func_x_minus_x_square(double x)
{
    return x*(1-x);
}

//Backward calculation for sigmoid
tensor* backward_propagation_sigmoid(const tensor* activation_input, tensor* gradient,tensor* output, activation* activation)
{
    mult_tensor_func(gradient, output, func_x_minus_x_square);
    return gradient;
}

//Buil sigmoid activation
activation* build_activation_sigmoid()
{
    activation* result = build_default_activation(SIGMOID);
    result->activation_backward_propagation=backward_propagation_sigmoid;
    result->activation_forward = activation_forward;
    result->activation_func=sigmoid;
    return result;
}