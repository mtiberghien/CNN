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

tensor* backward_propagation(tensor* activation_input, tensor* output_error, activation* activation)
{
    activation_input = activation->activation_func(activation_input);
    for(int i=0;i<output_error->size;i++)
    {
        output_error->v[i]*= activation_input->v[i];
    }
    return output_error;
}

activation* build_activation_relu(){
    activation* result = (activation*) malloc(sizeof(activation));
    result->backward_propagation=backward_propagation;
    result->activation_func=activation_func_relu;
    result->activation_func_prime=activation_func_prime_relu;
}

tensor* activation_func_softmax(tensor* input)
{
    double denominator = sum(input, exp);
    denominator = denominator == 0?1:denominator;
    for(int i=0;i<input->size;i++)
    {
        double d = input->v[i];
        input->v[i]=d/denominator;
    }
    return input;
}

tensor* activation_func_prime_softmax(tensor* activation_input)
{
    //TODO
    return activation_input;
}