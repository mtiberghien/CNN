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

//Clear memory required by training
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

//Clear memory required for training for layer with no activation
void clear_layer_training_memory_no_activation(layer *layer)
{
    #pragma omp parallel for
    for (int i = 0; i < layer->batch_size; i++)
    {
        clear_tensor(&layer->outputs[i]);
        clear_tensor(&layer->previous_gradients[i]);
    }
    free(layer->previous_gradients);
    free(layer->outputs);
    free(layer->layer_inputs);
}
//Clear layer memory required for prediction
void clear_layer_predict_memory(layer* layer)
{
    free_tensors(layer->outputs, layer->batch_size);
}
//Clear layer memory
void clear_layer(layer *layer)
{
    if(layer->parameters)
    {
        layer->clear_parameters(layer);
        free(layer->parameters);
    }
    if (layer->activation)
    {
        free(layer->activation);
    }
    free_shape(layer->input_shape);
    free_shape(layer->output_shape);
}
//Initialize memory required for training
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
//Initialize memory required for prediction
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
//Initialize memory required by training for layer without activation
void init_memory_training_no_activation(layer* layer)
{
    shape* input_shape = layer->input_shape;
    shape* output_shape = layer->output_shape;
    int batch_size = layer->batch_size;
    layer->layer_inputs=(tensor*) malloc(sizeof(tensor)*batch_size);
    layer->previous_gradients = (tensor *)malloc(sizeof(tensor)*batch_size);
    layer->outputs = malloc(sizeof(tensor) * batch_size);
    layer->activation_input = NULL;
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

//Default forward propagation loop when training
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
//Default forward propagation loop when predicting input batch
tensor *forward_propagation_predict_loop(const tensor *inputs, int batch_size, struct layer *layer, progression* progression)
{
    // Loop into input batch
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++)
    {
        //Output tensor memory allocation
        tensor *output = &layer->outputs[i];
        //Execute specific forward propagation
        layer->forward_calculation_predict(&inputs[i], output, layer);
        if(progression)
        {
            progression->next_step(progression);
        }
    }
    return layer->outputs;
}
//Predict the output according to an input array
tensor* layer_predict(tensor* inputs, int n_inputs, layer* layer)
{
    progression* progression = build_progression(n_inputs, "predicting");
    layer->batch_size=n_inputs;
    layer->init_predict_memory(layer);
    tensor* outputs = layer->forward_propagation_predict_loop(inputs, n_inputs, layer, progression);
    progression->done(progression);
    free_progression(progression);
    return outputs;
}

//Default save_parameters (do nothing)
void save_layer_parameters(FILE* fp, layer* layer)
{
}

//Default save_trainable_parameters (do nothing)
void save_trainable_parameters(FILE* fp, layer* layer)
{

}

void save_layer(FILE *fp, layer *layer)
{
    fprintf(fp, "input_shape:");
    save_shape(fp, layer->input_shape);
    fprintf(fp, ", output_shape:");
    save_shape(fp, layer->output_shape);
    fprintf(fp, ", type:%d\n", layer->type);
    fprintf(fp, "Parameters\n");
    layer->save_parameters(fp, layer);
    fprintf(fp, "Trainable parameters\n");
    layer->save_trainable_parameters(fp, layer);
    save_activation(fp, layer->activation);
}

char* to_string(layer* layer)
{
    switch(layer->type)
    {
        case FC: return "Dense_Layer";
        case CONV2D: return "Conv2D_Layer";
        case MAXPOOL2D: return "MaxPooling2D_Layer";
        case FLATTEN: return "Flatten_Layer";
        case PADDING2D: return "Padding2D_Layer";
        default: return "Layer";break;
    }
}

int get_layer_trainable_parameters_count(layer* layer)
{
    return 0;
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
    layer->read_trainable_parameters = read_trainable_parameters;
    layer->save_trainable_parameters = save_trainable_parameters;
    layer->get_trainable_parameters_count = get_layer_trainable_parameters_count;
    layer->to_string= to_string;
    layer->predict = layer_predict;
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
        case PADDING2D: configure_layer_Padding2D(layer); break;
        default: configure_layer_FC(layer);break;
    }
}

//Default read_parameters (do nothing)
void read_layer_parameters(FILE* fp, layer* layer)
{
}

//Default read_trainable_parameters (do nothing)
void read_trainable_parameters(FILE* fp, layer* layer)
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
        fscanf(fp, "Parameters\n");
        layer->read_parameters(fp, layer);
        layer->compile_layer(input_shape, layer);
        fscanf(fp, "Trainable parameters\n");
        layer->read_trainable_parameters(fp, layer);
        layer->activation = read_activation(fp);
        return layer;
    }
    free_shape(input_shape);
}