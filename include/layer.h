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
    tensor* activation_input;
    tensor* layer_inputs;
    //Stores the weight matrix as an array of tensor
    tensor* weights;
    //Stores the biases
    tensor biases;
    tensor* outputs;
    tensor* weights_gradients;
    tensor* previous_gradients;
    tensor biases_gradients;
    layer_type type;
    //Stores the batch size
    int batch_size;
    //Stores the layer input size (number of elements of an input tensor)
    int input_size;
    //Stores the layer output size (number of elements of an output tensor)
    int output_size;
    //Stores the output tensor batch (after activation)
    void (*compile_layer)(int input_size, struct layer* layer);
    void (*init_training_memory)(struct layer* layer);
    void (*init_predict_memory)(struct layer* layer);
    void (*clear_predict_memory)(struct layer* layer);
    void (*clear_training_memory)(struct layer* layer);
    //Stores the forward propagation loop
    tensor* (*forward_propagation_training_loop)(const tensor* inputs, int batch_size, struct layer* layer, progression* progression);
    tensor* (*forward_propagation_predict_loop)(const tensor* inputs, int batch_size, struct layer* layer, progression* progression);
    //Stores the backward propagation loop
    tensor* (*backward_propagation_loop)(tensor* gradients, optimizer* optimizer, struct layer* layer, int layer_index);
    //Stores the specific forward propagation calculation (transition from inputs to outputs with activation)
    tensor* (*forward_calculation_training)(const tensor* input, tensor* output, tensor* activation_input, struct layer* layer);
    tensor* (*forward_calculation_predict)(const tensor* input, tensor* output, struct layer* layer);
    //Stores the specific backward propagation calculation (transition from gradient of next layer to gradient of current layer and update of weights and biases) 
    void (*backward_calculation)(tensor* biases_gradient, tensor* weights_gradient, optimizer* optimizer, struct layer* layer, int layer_index);
    //Stores the activation object
    activation* activation;
} layer;

void clear_layer(layer*);
layer* build_layer(layer_type type, int output_size, activation* activation);
void save_layer(FILE *fp, layer* layer);
layer* read_layer(FILE *fp);
#endif