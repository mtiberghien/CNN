#ifndef CNN_LAYER
#define CNN_LAYER

#include "tensor.h"
#include "optimizer.h"
#include "activation.h"
#include "progression.h"
#include "common.h"
#include <stdio.h>

typedef enum layer_type{FC,CONV2D, MAXPOOL2D, FLATTEN, PADDING2D} layer_type;

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
    void (*save_trainable_parameters)(FILE*, struct layer* layer);
    void (*read_trainable_parameters)(FILE*, struct layer* layer);
    void (*build_shape_list)(struct layer* layer, shape_list* shape_list);
    char* (*to_string)(struct layer* layer);
    int (*get_trainable_parameters_count)(struct layer* layer);
    tensor* (*predict)(tensor* intputs, int n_inputs, struct layer* layer);
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
void save_layer(FILE *fp, layer* layer);
void save_trainable_parameters(FILE* fp, layer* layer);
layer* read_layer(FILE *fp);
void read_trainable_parameters(FILE* fp, layer* layer);
layer* build_layer_FC(int output_size, activation* activation);
layer* build_layer_Conv2D(int output_channel_size, int kernel_width, int kernel_height, int stride, short padding, activation* activation);
layer* build_layer_Flatten();
layer* build_layer_MaxPooling2D(int pool_height, int pool_width, int stride);
layer* build_layer_Padding2D(int padding_height, int padding_width);
void configure_layer_Conv2D(layer* layer);
void configure_layer_FC(layer* layer);
void configure_layer_Flatten(layer* layer);
void configure_layer_MaxPooling2D(layer* layer);
void configure_layer_Padding2D(layer* layer);
void init_memory_training(layer* layer);
void init_memory_predict(layer* layer);
void clear_layer_predict_memory(layer* layer);
void clear_layer_training_memory(layer *layer);
tensor *forward_propagation_training_loop(const tensor *inputs, int batch_size, struct layer *layer, progression* progression);
tensor *forward_propagation_predict_loop(const tensor *inputs, int batch_size, struct layer *layer, progression* progression);
void configure_default_layer(layer* layer);
void clear_layer_parameters(struct layer* layer);
void build_layer_shape_list(layer* layer, shape_list* shape_list);
void save_layer_parameters(FILE*, struct layer* layer);
void read_layer_parameters(FILE*, struct layer* layer);
void clear_layer_training_memory_no_activation(layer *layer);
void init_memory_training_no_activation(layer* layer);
tensor *forward_propagation_training_loop_no_activation(const tensor *inputs, int batch_size, struct layer *layer, progression* progression);
#endif