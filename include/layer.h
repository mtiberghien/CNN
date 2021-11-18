#ifndef CNN_LAYER
#define CNN_LAYER

#include "tensor.h"
#include "optimizer.h"
#include "activation.h"

//Represent a layer of a sequential neural network
typedef struct layer{
    tensor* mean_activation_input;
    tensor* mean_output;
    tensor* mean_input;
    //Stores the batch size
    int batch_size;
    short is_training;
    //Stores the weight matrix as a single tensor
    tensor weights;
    //Stores the biases
    tensor biases;
    //Stores the layer input size (number of elements of an input tensor)
    int input_size;
    //Stores the layer output size (number of elements of an output tensor)
    int output_size;
    //Stores the output tensor batch (after activation)
    tensor* outputs;
    //Stores the forward propagation loop
    tensor* (*forward_propagation_loop)(tensor* inputs, int batch_size, short is_training, struct layer* layer);
    //Stores the backward propagation loop
    tensor* (*backward_propagation)(tensor* gradients, optimizer* optimizer, struct layer* layer);
    //Stores the specific forward propagation calculation (transition from inputs to outputs with activation)
    tensor* (*forward_calculation)(tensor* input, tensor* output, struct layer* layer);
    //Stores the specific backward propagation calculation (transition from gradient of next layer to gradient of current layer and update of weights and biases) 
    tensor* (*backward_calculation)(tensor* gradient, tensor* gradient_previous, optimizer* optimizer, struct layer* layer);
    //Stores the activation object
    activation* activation;
} layer;

void clear_layer(layer*);
void clear_layer_input_output(layer*);
tensor* forward_propagation_loop(tensor* inputs, int batch_size, short is_training, struct layer* layer);
tensor* backward_propagation(tensor* gradients, optimizer* optimizer, struct layer* layer);
layer* build_layer_FC(int input_size, int output_size, activation* activation);
tensor* forward_calculation_FC(tensor* input, tensor* output, layer* layer);
tensor* backward_calculation_FC(tensor* gradient, tensor* gradient_previous, optimizer* optimizer, layer* layer);

#endif