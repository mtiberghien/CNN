#include "include/activation.h"
#include "include/tensor.h"
#include "math.h"

//Relu activation calculation
tensor* activation_func_relu(tensor* input)
{
    for(int i=0;i<input->size;i++)
    {
        double d = input->v[i];
        input->v[i]=d<0?0:d;
    }
    return input;
}

//Relu derivative calculation
tensor* activation_func_prime_relu(tensor* activation_input)
{
    for(int i=0;i<activation_input->size;i++)
    {
        double d = activation_input->v[i];
        activation_input->v[i]=d<=0?0:1;
    }
    return activation_input;
}

tensor* activation_backward_propagation(tensor* activation_input, tensor* gradient, tensor* output, activation* activation)
{
    activation_input = activation->activation_func_prime(activation_input);
    for(int i=0;i<gradient->size;i++)
    {
        gradient->v[i]*= activation_input->v[i];
    }
    return gradient;
}

activation* build_activation_relu()
{
    activation* result = (activation*) malloc(sizeof(activation));
    result->type = RELU;
    result->activation_backward_propagation=activation_backward_propagation;
    result->activation_func=activation_func_relu;
    result->activation_func_prime=activation_func_prime_relu;
    return result;
}

tensor* activation_func_softmax(tensor* input)
{
    double max_value = max(input);
    input = sub(input, max_value);
    double denominator = sum(input, exp);
    double invert_denominator = denominator == 0?1:(double)1.0/denominator;
    for(int i=0;i<input->size;i++)
    {
        double d = exp(input->v[i]);
        input->v[i]=d*invert_denominator;
    }
    return input;
}

tensor* backward_propagation_softmax(tensor* activation_input, tensor* gradient, tensor* output, activation* activation)
{
    tensor gradient_product;
    initialize_tensor(&gradient_product, gradient->size);
    double sum =0;
    for(int i=0;i<gradient->size;i++)
    {
        gradient_product.v[i]=gradient->v[i]*output->v[i];
        sum+=gradient_product.v[i];
    }
    for(int i=0;i<gradient->size;i++)
    {
        gradient->v[i]-=sum;
        gradient->v[i]*=output->v[i];
    }
    free(gradient_product.v);
    return gradient;
}

activation* build_activation_softmax()
{
    activation* result = (activation*) malloc(sizeof(activation));
    result->type = SOFTMAX;
    result->activation_backward_propagation=backward_propagation_softmax;
    result->activation_func=activation_func_softmax;
    return result;
}