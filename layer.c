#include "include/layer.h"
#include "include/tensor.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

//Default empty shape_list
void build_layer_shape_list(layer* layer, shape_list* shape_list)
{
    shape_list->n_shapes=0;
    shape_list->shapes=NULL;
}

//Default clear_parameters (do nothing)
void clear_layer_parameters(layer* layer)
{
}

//Clear memory of temporary stored inputs and outputs
void clear_layer_training_memory(layer *layer)
{
    #pragma omp parallel for
    for (int i = 0; i < layer->batch_size; i++)
    {
        clear_tensor(&layer->outputs[i]);
        clear_tensor(&layer->activation_input[i]);
        clear_tensor(&layer->previous_gradients[i]);
    }
    free(layer->previous_gradients);
    free(layer->activation_input);
    free(layer->layer_inputs);
    free(layer->outputs);
}

//Clear training memory when no activation
void clear_layer_training_memory_no_activation(layer *layer)
{
    #pragma omp parallel for
    for (int i = 0; i < layer->batch_size; i++)
    {
        clear_tensor(&layer->outputs[i]);
        clear_tensor(&layer->previous_gradients[i]);
    }
    free(layer->previous_gradients);
    free(layer->layer_inputs);
    free(layer->outputs);
}

void clear_layer_predict_memory(layer* layer)
{
    #pragma omp parallel for
    for (int i = 0; i < layer->batch_size; i++)
    {
        clear_tensor(&layer->outputs[i]);
    }
    free(layer->outputs);
}

void clear_layer(layer *layer)
{
    layer->clear_parameters(layer);
    if (layer->activation)
    {
        free(layer->activation);
    }
    clear_shape(layer->input_shape);
    clear_shape(layer->output_shape);
    free(layer->output_shape);
    free(layer->input_shape);
}

void init_memory_training(layer* layer)
{
    shape* input_shape = layer->input_shape;
    shape* output_shape = layer->output_shape;
    int batch_size = layer->batch_size;
    layer->layer_inputs=(tensor*) malloc(sizeof(tensor)*batch_size);
    layer->activation_input = (tensor *)malloc(sizeof(tensor)*batch_size);
    layer->previous_gradients = (tensor *)malloc(sizeof(tensor)*batch_size);
    layer->outputs = malloc(sizeof(tensor) * batch_size);
    #pragma omp parallel for
    for(int i=0;i<batch_size;i++)
    {
        initialize_tensor(&layer->outputs[i], output_shape);
        initialize_tensor(&layer->activation_input[i], output_shape);
        initialize_tensor(&layer->previous_gradients[i], input_shape);
    }
}

void init_memory_predict(layer* layer)
{
    int batch_size = layer->batch_size;
    shape* output_shape = layer->output_shape;
    layer->outputs = malloc(sizeof(tensor) * batch_size);
    #pragma omp parallel for
    for(int i=0;i<batch_size;i++)
    {
        initialize_tensor(&layer->outputs[i], output_shape);
    }
}

void init_memory_training_no_activation(layer* layer)
{
    shape* input_shape = layer->input_shape;
    shape* output_shape = layer->output_shape;
    int batch_size = layer->batch_size;
    layer->layer_inputs=(tensor*) malloc(sizeof(tensor)*batch_size);
    layer->activation_input = NULL;
    layer->previous_gradients = (tensor *)malloc(sizeof(tensor)*batch_size);
    layer->outputs = malloc(sizeof(tensor) * batch_size);
    #pragma omp parallel for
    for(int i=0;i<batch_size;i++)
    {
        initialize_tensor(&layer->outputs[i], output_shape);
        initialize_tensor(&layer->previous_gradients[i], input_shape);
    }
}

//Forward propagation loop for layer with no activation
tensor *forward_propagation_training_loop_no_activation(const tensor *inputs, int batch_size, struct layer *layer, progression* progression)
{
    // Loop into input batch
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++)
    {
        //Output tensor memory allocation
        tensor *output = &layer->outputs[i];
        const tensor* input = &inputs[i];
        layer->layer_inputs[i]=inputs[i];
        //Execute specific forward propagation
        layer->forward_calculation_training(input, output, NULL, layer);
    }
    return layer->outputs;
}

//Default forward propagation loop
tensor *forward_propagation_training_loop(const tensor *inputs, int batch_size, struct layer *layer, progression* progression)
{
    // Loop into input batch
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++)
    {
        //Output tensor memory allocation
        tensor *output = &layer->outputs[i];
        tensor* activation_input = &layer->activation_input[i];
        const tensor* input = &inputs[i];
        layer->layer_inputs[i]=inputs[i];
        //Execute specific forward propagation
        layer->forward_calculation_training(input, output, activation_input, layer);
    }
    return layer->outputs;
}

tensor *forward_propagation_predict_loop(const tensor *inputs, int batch_size, struct layer *layer, progression* progression)
{
    // Loop into input batch
    #pragma omp parallel for shared(progression)
    for (int i = 0; i < batch_size; i++)
    {
        //Output tensor memory allocation
        tensor *output = &layer->outputs[i];
        //Execute specific forward propagation
        layer->forward_calculation_predict(&inputs[i], output, layer);
        if(progression)
        {
            progression->call_back(progression);
        }
    }
    return layer->outputs;
}

//Default save_parameters (do nothing)
void save_layer_parameters(FILE* fp, layer* layer)
{
}

void save_layer(FILE *fp, layer *layer)
{
    fprintf(fp, "input_shape:");
    save_shape(fp, layer->input_shape);
    fprintf(fp, ", output_shape:");
    save_shape(fp, layer->output_shape);
    fprintf(fp, ", type:%d\n", layer->type);
    layer->save_parameters(fp, layer);
    save_activation(fp, layer->activation);
}





void configure_default_layer(layer* layer)
{
    layer->forward_propagation_training_loop = forward_propagation_training_loop;
    layer->forward_propagation_predict_loop = forward_propagation_predict_loop;
    layer->init_predict_memory = init_memory_predict;
    layer->init_training_memory = init_memory_training;
    layer->clear_training_memory= clear_layer_training_memory;
    layer->clear_predict_memory = clear_layer_predict_memory;
    layer->clear_parameters = clear_layer_parameters;
    layer->save_parameters = save_layer_parameters;
    layer->read_parameters = read_layer_parameters;
    layer->build_shape_list = build_layer_shape_list;
}

layer* build_layer(layer_type type, shape* output_shape)
{
    layer *layer = (struct layer *)malloc(sizeof(struct layer));
    layer->type = type;
    layer->output_shape = output_shape;
    switch(type)
    {
        case CONV2D: configure_layer_Conv2D(layer);break;
        case MAXPOOL2D: configure_layer_MaxPooling2D(layer);break;
        case FLATTEN: configure_layer_Flatten(layer);break;
        default: configure_layer_FC(layer);break;
    }
}

//Default read_parameters (do nothing)
void read_layer_parameters(FILE* fp, layer* layer)
{
}

layer *read_layer(FILE *fp)
{
    int type;
    int input_dimension, output_dimension;
    fscanf(fp, "input_shape:");
    shape* input_shape = read_shape(fp);
    fscanf(fp, ", output_shape:");
    shape* output_shape = read_shape(fp);
    fscanf(fp, ", type:%d\n", &type);
    if (type >= 0)
    {
        layer *layer = build_layer(type, output_shape);
        layer->compile_layer(input_shape, layer);
        layer->read_parameters(fp, layer);
        layer->activation = read_activation(fp);
        return layer;
    }
    clear_shape(input_shape);
    free(input_shape);
}