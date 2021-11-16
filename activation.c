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
tensor* activation_func_prime_relu(tensor* output_error)
{
    for(int i=0;i<output_error->size;i++)
    {
        double d = output_error->v[i];
        output_error->v[i]=d<=0?0:1;
    }
    return output_error;
}

activation* build_activation_relu(){
    activation* result = (activation*) malloc(sizeof(activation));
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

tensor* activation_func_prime_softmax(tensor* output_error)
{
    //TODO
    return output_error;
}