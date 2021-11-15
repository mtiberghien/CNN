#ifndef CNN_LAYER
#define CNN_LAYER

#include "tensor.h"
#include "optimizer.h"

typedef struct layer{
    tensor* input;
    int n_inputs;
    tensor weights;
    tensor biases;
    int input_size;
    int output_size;
    tensor* output;
    tensor* (*forward_propagation)(tensor* input, int n_inputs, struct layer layer);
    tensor* (*backward_propagation)(tensor* output_error, optimizer optimizer, struct layer layer);
} layer;

void clear_output(layer);

layer build_layer_FC(int input_size, int output_size);
tensor* forward_propagation_FC(tensor* input, int n_inputs, layer layer);
tensor* backward_propagation_FC(tensor* output_error, optimizer optimizer, layer layer);

#endif