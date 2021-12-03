#ifndef CNN_LAYER
#define CNN_LAYER

#include "tensor.h"
#include "optimizer.h"
#include "activation.h"
#include "progression.h"
#include "common.h"
#include <stdio.h>

typedef enum layer_type{FC} layer_type;

//Represent a layer of a sequential neural network
typedef struct layer{
    tensor* activation_input;
    tensor* layer_inputs;
    tensor* outputs;
    tensor* previous_gradients;
    layer_type type;
    //Stores the batch size
    int batch_size;
    //Stores the layer input size (number of elements of an input tensor)
    shape* input_shape;
    //Stores the layer output size (number of elements of an output tensor)
    shape* output_shape;
    void* parameters;
    //Stores the output tensor batch (after activation)
    void (*compile_layer)(shape* input_shape, struct layer* layer);
    void (*init_training_memory)(struct layer* layer);
    void (*init_predict_memory)(struct layer* layer);
    void (*clear_predict_memory)(struct layer* layer);
    void (*clear_training_memory)(struct layer* layer);
    void (*clear_parameters)(struct layer* layer);
    void (*save_parameters)(FILE*, struct layer* layer);
    void (*read_parameters)(FILE*, struct layer* layer);
    //Stores the forward propagation loop
    tensor* (*forward_propagation_training_loop)(const tensor* inputs, int batch_size, struct layer* layer, progression* progression);
    tensor* (*forward_propagation_predict_loop)(const tensor* inputs, int batch_size, struct layer* layer, progression* progression);
    //Stores the backward propagation loop
    tensor* (*backward_propagation_loop)(tensor* gradients, optimizer* optimizer, struct layer* layer, int layer_index);
    //Stores the specific forward propagation calculation (transition from inputs to outputs with activation)
    tensor* (*forward_calculation_training)(const tensor* input, tensor* output, tensor* activation_input, struct layer* layer);
    tensor* (*forward_calculation_predict)(const tensor* input, tensor* output, struct layer* layer);
    //Stores the specific backward propagation calculation (transition from gradient of next layer to gradient of current layer and update of weights and biases) 
    void (*backward_calculation)(optimizer* optimizer, struct layer* layer, int layer_index);
    //Stores the activation object
    activation* activation;
} layer;

void clear_layer(layer*);
layer* build_layer_FC(int output_size, activation* activation);
void save_layer(FILE *fp, layer* layer);
layer* read_layer(FILE *fp);
#endif