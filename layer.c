#include "include/layer.h"
#include "include/tensor.h"
#include <stdlib.h>

void clear_output(layer layer){
    for(int i=0;i<layer.n_inputs;i++)
    {
        clear_tensor(layer.output[i]);
    }
    free(layer.output);
}

layer buildFCLayer(int input_size, int output_size){
    layer layer;
    layer.input_size = input_size;
    layer.output_size = output_size;
    layer.weights.size = input_size*output_size;
    for(int i=0;i<input_size*output_size;i++){
        layer.weights.v[i] = (double)rand() / (double)RAND_MAX ;
    }
    layer.bias.size = output_size;
    layer.bias.v = calloc(output_size,sizeof(double));
    layer.forward_propagation=FC_forward_propagation;
    layer.backward_propagation=FC_backward_propagation;
    return layer;
}

tensor* FC_forward_propagation(tensor* input, int n_inputs, layer layer){
    layer.input = input;
    layer.n_inputs = n_inputs;
    clear_output(layer);
    layer.output=malloc(sizeof(tensor)*n_inputs);
    for(int i=0;i<n_inputs;i++)
    {
        tensor t = layer.output[i];
        t.size = layer.output_size;
        t.v = calloc(layer.output_size,sizeof(double));
        for(int j=0;j<layer.output_size;j++){
            for(int k=0;k<layer.input_size;k++){
                t.v[j] += layer.weights.v[j*layer.input_size+k]* (input[i].v[k]);
            }
            t.v[j] += + layer.bias.v[j];
        }
    }
    return layer.output;
}

tensor* FC_backward_propagation(tensor* output_error, optimizer optimizer, layer layer)
{
    tensor* input_error = (tensor*)malloc(sizeof(tensor)*layer.n_inputs);
    for(int i=0;i<layer.n_inputs;i++)
    {
        tensor ie_t = input_error[i];
        ie_t.size = layer.input_size;
        ie_t.v = calloc(layer.input_size, sizeof(double));
        for(int j=0;j<layer.input_size;j++){
            for(int k=0;k<layer.output_size;k++){
                ie_t.v[j] += layer.weights.v[j+layer.input_size*k]*(output_error[i].v[k]);
            }
        }
    }
    //TODO Finish weight and biases update
    return input_error;
}