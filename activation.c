#include "include/activation.h"
#include "include/tensor.h"
#include "math.h"

double relu(double x)
{
    return x<0?0:x;
}

double relu_prime(double x)
{
    return x<=0?0:1;
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

tensor* activation_backward_propagation(const tensor* activation_input, tensor* gradient, tensor* output, activation* activation)
{
    mult_tensor_func(gradient, activation_input, activation->activation_func_prime);
    return gradient;
}

tensor* backward_propagation_sigmoid(const tensor* activation_input, tensor* gradient,tensor* output, activation* activation)
{
    mult_tensor_func(gradient, output, func_x_minus_x_square);
    return gradient;
}

tensor* activation_forward_softmax(tensor* input, activation* activation)
{
    double max_value = max(input);
    input = sub(input, max_value);
    double denominator = sum(input, exp);
    double invert_denominator = denominator == 0?1:(double)1.0/denominator;
    int* iterator = get_iterator(input);
    while(!input->is_done(input, iterator))
    {
        double v = input->get_value(input, iterator);
        double d = exp(v);
        input->set_value(input, iterator, d*invert_denominator);
        iterator = input->get_next(input, iterator);
    }
    free(iterator);
    return input;
}

tensor* backward_propagation_softmax(const tensor* activation_input, tensor* gradient, tensor* output, activation* activation)
{
    tensor gradient_product;
    initialize_tensor(&gradient_product, gradient->shape);
    double sum =0;
    int* iterator = get_iterator(gradient);
    while(!gradient->is_done(gradient, iterator))
    {
        double output_v = output->get_value(output, iterator);
        double gradient_v = gradient->get_value(gradient, iterator);
        double product = gradient_v*output_v;
        gradient_product.set_value(&gradient_product, iterator, product);
        sum+= product;
        iterator = gradient->get_next(gradient, iterator);
    }
    free(iterator);
    iterator = get_iterator(gradient);
    while(!gradient->is_done(gradient, iterator))
    {
        double output_v = output->get_value(output, iterator);
        double gradient_v = gradient->get_value(gradient, iterator);
        gradient->set_value(gradient, iterator,(gradient_v - sum)*output_v);
        iterator = gradient->get_next(gradient, iterator);
    }
    free(iterator);
    clear_tensor(&gradient_product);
    return gradient;
}

activation* build_default_activation(activation_type type)
{
    activation* result = (activation*) malloc(sizeof(activation));
    result->type = type;
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

activation* build_activation_softmax()
{
    activation* result = build_default_activation(SOFTMAX);
    result->activation_backward_propagation=backward_propagation_softmax;
    result->activation_forward= activation_forward_softmax;
    return result;
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

activation* build_activation_sigmoid()
{
    activation* result = build_default_activation(SIGMOID);
    result->activation_backward_propagation=backward_propagation_sigmoid;
    result->activation_forward = activation_forward;
    result->activation_func=sigmoid;
    return result;
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