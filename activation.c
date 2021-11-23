#include "include/activation.h"
#include "include/tensor.h"
#include "math.h"

double relu(double x)
{
    return x<0?0:x;
}

double relu_prime(double x)
{
    return x<0?0:1;
}


double tanh_prime(double x)
{
    return 1-pow(tanh(x),(double)2);
}

//Relu activation calculation
tensor* activation_func_relu(tensor* input)
{
    apply_func(input, relu);
    return input;
}

//Relu derivative calculation
tensor* activation_func_prime_relu(tensor* activation_input)
{
    apply_func(activation_input, relu_prime);
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

activation* build_activation(activation_type type)
{
    switch(type){
        case RELU: return build_activation_relu();
        case SOFTMAX: return build_activation_softmax();
        case TANH: return build_activation_tanh();
        default: return NULL;
    }
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

tensor* activation_func_tanh(tensor* input)
{
    apply_func(input, tanh);
    return input;
}

tensor* activation_func_prime_tanh(tensor* activation_input)
{
    apply_func(activation_input, tanh_prime);
    return activation_input;
}

activation* build_activation_tanh()
{
    activation* result = (activation*) malloc(sizeof(activation));
    result->type = TANH;
    result->activation_backward_propagation=activation_backward_propagation;
    result->activation_func=activation_func_tanh;
    result->activation_func_prime=activation_func_prime_tanh;
    return result;
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