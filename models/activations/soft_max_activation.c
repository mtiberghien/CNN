#include "../../include/activation.h"
#include <math.h>

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

activation* build_activation_softmax()
{
    activation* result = build_default_activation(SOFTMAX);
    result->activation_backward_propagation=backward_propagation_softmax;
    result->activation_forward= activation_forward_softmax;
    return result;
}

