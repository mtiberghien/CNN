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

double sigmoid(double x)
{
    return 1/(1+exp(-x));
}

double func_x_minus_x_square(double x)
{
    return x*(1-x);
}

tensor* activation_forward(tensor* input, activation* activation)
{
    apply_func(input, activation->activation_func);
    return input;
}

tensor* activation_backward_propagation(tensor* activation_input, tensor* gradient, tensor* output, activation* activation)
{
    mult_tensor_func(gradient, activation_input, activation->activation_func_prime);
    return gradient;
}

activation* build_activation_relu()
{
    activation* result = (activation*) malloc(sizeof(activation));
    result->type = RELU;
    result->activation_backward_propagation=activation_backward_propagation;
    result->activation_forward = activation_forward;
    result->activation_func=relu;
    result->activation_func_prime=relu_prime;
    return result;
}

activation* build_activation_sigmoid()
{
    activation* result = (activation*) malloc(sizeof(activation));
    result->type = SIGMOID;
    result->activation_backward_propagation=backward_propagation_sigmoid;
    result->activation_forward = activation_forward;
    result->activation_func=sigmoid;
    return result;
}
tensor* backward_propagation_sigmoid(tensor* activation_input, tensor* gradient,tensor* output, activation* activation)
{
    mult_tensor_func(gradient, output, func_x_minus_x_square);
    return gradient;
}

activation* build_activation(activation_type type)
{
    switch(type){
        case RELU: return build_activation_relu();
        case SOFTMAX: return build_activation_softmax();
        case TANH: return build_activation_tanh();
        case SIGMOID: return build_activation_sigmoid();
        default: return NULL;
    }
}

tensor* activation_forward_softmax(tensor* input, activation* activation)
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

activation* build_activation_tanh()
{
    activation* result = (activation*) malloc(sizeof(activation));
    result->type = TANH;
    result->activation_backward_propagation=activation_backward_propagation;
    result->activation_forward = activation_forward;
    result->activation_func=tanh;
    result->activation_func_prime=tanh_prime;
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
    result->activation_forward= activation_forward_softmax;
    return result;
}

void save_activation(FILE* fp, activation* activation)
{
    fprintf(fp, "activation:%d\n", activation!=NULL? activation->type:-1);
}

activation* read_activation(FILE* fp)
{
    int type;
    fscanf(fp, "activation:%d\n", &type);
    if(type>=0)
    {
        return build_activation(type);
    }
    return NULL;
}