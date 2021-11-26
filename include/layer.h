#ifndef CNN_LAYER
#define CNN_LAYER

#include "tensor.h"
#include "optimizer.h"
#include "activation.h"
#include "progression.h"
#include <stdio.h>

typedef enum layer_type{FC} layer_type;

//Represent a layer of a sequential neural network
typedef struct layer{
    layer_type type;
    tensor* activation_input;
    tensor* layer_input;
    //Stores the batch size
    int batch_size;
    short is_training;
    //Stores the weight matrix as an array of tensor
    tensor* weights;
    //Stores the biases
    tensor biases;
    //Stores the layer input size (number of elements of an input tensor)
    int input_size;
    //Stores the layer output size (number of elements of an output tensor)
    int output_size;
    //Stores the output tensor batch (after activation)
    tensor* outputs;
    void (*compile_layer)(int input_size, struct layer* layer);
    //Stores the forward propagation loop
    tensor* (*forward_propagation_loop)(const tensor* inputs, int batch_size, short is_training, struct layer* layer, progression* progression);
    //Stores the backward propagation loop
    tensor* (*backward_propagation_loop)(tensor* gradients, optimizer* optimizer, struct layer* layer, int layer_index);
    //Stores the specific forward propagation calculation (transition from inputs to outputs with activation)
    tensor* (*forward_calculation)(const tensor* input, tensor* output, tensor* activation_input, struct layer* layer);
    //Stores the specific backward propagation calculation (transition from gradient of next layer to gradient of current layer and update of weights and biases) 
    void (*backward_calculation)(tensor* biases_gradient, tensor* weights_gradient, optimizer* optimizer, struct layer* layer, int layer_index);
    //Stores the activation object
    activation* activation;
} layer;

void clear_layer(layer*);
void clear_layer_backpropagation_memory(layer*, tensor* weights_gradient, tensor* biases_gradient, tensor* gradients);
void compile_layer(int input_size, layer*);
tensor* forward_propagation_loop(const tensor* inputs, int batch_size, short is_training, struct layer* layer, progression* progression);
tensor* backward_propagation_loop(tensor* gradients, optimizer* optimizer, struct layer* layer, int layer_index);
layer* build_layer(layer_type type, int output_size, activation* activation);
void build_layer_FC(layer* layer);
tensor* forward_calculation_FC(const tensor* input, tensor* output, tensor* activation_input, layer* layer);
void backward_calculation_FC(tensor* biases_gradient, tensor* weights_gradient, optimizer* optimizer, layer* layer, int layer_index);
void save_layer(FILE *fp, layer* layer);
layer* read_layer(FILE *fp);
#endif