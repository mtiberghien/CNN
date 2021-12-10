#include "include/activation.h"
#include "include/tensor.h"
#include "math.h"

//Default activation forward calculation
tensor* activation_forward(tensor* input, activation* activation)
{
    apply_func(input, activation->activation_func);
    return input;
}
//Default activation backward calculation
tensor* activation_backward_propagation(const tensor* activation_input, tensor* gradient, tensor* output, activation* activation)
{
    mult_tensor_func(gradient, activation_input, activation->activation_func_prime);
    return gradient;
}
//Default activation build task
activation* build_default_activation(activation_type type)
{
    activation* result = (activation*) malloc(sizeof(activation));
    result->type = type;
}
//Build activation according to the provided type
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
//Write activation to file
void save_activation(FILE* fp, activation* activation)
{
    fprintf(fp, "activation:%d\n", activation!=NULL? activation->type:-1);
}
//Read activation from file
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