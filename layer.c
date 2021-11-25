#include "include/layer.h"
#include "include/tensor.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

//Clear memory of temporary stored inputs and outputs
void clear_layer_input_output(layer *layer)
{
    clear_tensor(layer->mean_input);
    clear_tensor(layer->mean_activation_input);
    clear_tensor(layer->mean_output);
    for (int i = 0; i < layer->batch_size; i++)
    {
        clear_tensor(&layer->outputs[i]);
    }
    free(layer->mean_activation_input);
    free(layer->mean_input);
    free(layer->mean_output);
    free(layer->outputs);
}

void clear_layer(layer *layer)
{
    clear_tensors(layer->weights, layer->output_size);
    free(layer->weights);
    clear_tensor(&layer->biases);
    if (layer->activation)
    {
        free(layer->activation);
    }
}

void compile_layer(int input_size, layer *layer)
{
    layer->input_size = input_size;
    layer->weights = (tensor *)malloc(layer->output_size * sizeof(tensor));
    double invert_rand_max = (double)1.0 / (double)RAND_MAX;
    double limit = sqrt((double)6 / (input_size + layer->output_size));
    for (int i = 0; i < layer->output_size; i++)
    {
        initialize_tensor(&layer->weights[i], input_size);
        for (int j = 0; j < input_size; j++)
        {
            layer->weights[i].v[j] = (2 * limit * ((double)rand() * invert_rand_max)) - limit;
        }
    }
    //Initialize biases
    initialize_tensor(&layer->biases, layer->output_size);
    for (int i = 0; i < layer->output_size; i++)
    {
        layer->biases.v[i] = (2 * limit * ((double)rand() * invert_rand_max)) - limit;
    }
}

layer *build_layer(layer_type type, int output_size, activation *activation)
{
    layer *layer = (struct layer *)malloc(sizeof(struct layer));
    layer->type = type;
    //Store input and output size
    layer->output_size = output_size;
    //Store activation function
    layer->activation = activation;
    switch (type)
    {
    default:
        build_layer_FC(layer);
        break;
    }
    return layer;
}

void build_layer_FC(layer *layer)
{

    //Set used methods for the layer
    layer->compile_layer = compile_layer;
    layer->forward_propagation_loop = forward_propagation_loop;
    layer->backward_propagation = backward_propagation;
    layer->forward_calculation = forward_calculation_FC;
    layer->backward_calculation = backward_calculation_FC;
}

//Common forward propagation loop
tensor *forward_propagation_loop(const tensor *inputs, int batch_size, double invert_batch_size, short is_training, struct layer *layer)
{
    int output_size = layer->output_size;
    layer->is_training = is_training;
    if (is_training)
    {
        layer->mean_activation_input = (tensor *)malloc(sizeof(tensor));
        layer->mean_output = (tensor *)malloc(sizeof(tensor));
        layer->invert_output_size = (double)1.0 / output_size;
        layer->mean_input = (tensor *)malloc(sizeof(tensor));
        initialize_tensor(layer->mean_activation_input, output_size);
        initialize_tensor(layer->mean_output, output_size);
        initialize_tensor(layer->mean_input, layer->input_size);
        //Store inputs (used in backward propagation)
        layer->batch_size = batch_size;
    }

    //Allocate memory for outputs and activation_input
    layer->outputs = malloc(sizeof(tensor) * batch_size);
    // Loop into input batch
    for (int i = 0; i < batch_size; i++)
    {
        //Output tensor memory allocation
        tensor *output = &layer->outputs[i];
        initialize_tensor(output, output_size);
        if (is_training)
        {
            for (int j = 0; j < layer->input_size; j++)
            {
                layer->mean_input->v[j] += (inputs[i].v[j] * invert_batch_size);
            }
        }
        //Execute specific forward propagation
        layer->forward_calculation(&inputs[i], output, invert_batch_size, layer);
    }
    return layer->outputs;
}

//Common backward propagation loop
tensor *backward_propagation(tensor *mean_gradient, optimizer *optimizer, struct layer *layer, int layer_index)
{
    //Previous layer gradient error tensor memory allocation
    tensor *gradient_previous = (tensor *)malloc(sizeof(tensor));
    initialize_tensor(gradient_previous, layer->input_size);
    layer->backward_calculation(mean_gradient, gradient_previous, optimizer, layer, layer_index);
    if (layer->is_training)
    {
        clear_tensor(mean_gradient);
        free(mean_gradient);
        clear_layer_input_output(layer);
    }
    return gradient_previous;
}

//Forward propagation function for Fully Connected layer (perceptron)
tensor *forward_calculation_FC(const tensor *input, tensor *output, double invert_batch_size, layer *layer)
{
    //Loop into output tensor
    for (int i = 0; i < layer->output_size; i++)
    {
        //Loop into input tensor
        for (int j = 0; j < layer->input_size; j++)
        {
            //sum weighted input element using weights matrix
            output->v[i] += layer->weights[i].v[j] * (input->v[j]);
        }
        //Add bias
        output->v[i] += layer->biases.v[i];
        if (layer->is_training)
        {
            //Store the activation input
            layer->mean_activation_input->v[i] += (output->v[i] * invert_batch_size);
        }
    }
    if (layer->activation)
    {
        //Execute activation function and return output tensor
        output = layer->activation->activation_forward(output, layer->activation);
    }
    if (layer->is_training)
    {
        for (int j = 0; j < layer->output_size; j++)
        {
            layer->mean_output->v[j] += (output->v[j] * invert_batch_size);
        }
    }
}

//Backward propagation function for Fully Connected layer (perceptron)
tensor *backward_calculation_FC(tensor *gradient, tensor *gradient_previous, optimizer *optimizer, layer *layer, int layer_index)
{
    if (layer->activation)
    {
        //Back propagate the gradient error tensor
        gradient = layer->activation->activation_backward_propagation(layer->mean_activation_input, gradient, layer->mean_output, layer->activation);
    }
    //Update biases using new gradient
    for (int i = 0; i < layer->output_size; i++)
    {
        double gradient_i = gradient->v[i];
        //Optimizer update bias using the activation primitive
        layer->biases.v[i] = optimizer->apply_gradient(layer->biases.v[i], gradient_i, layer_index, i, optimizer);
        //Calculate the gradient for previous layer and update weights
        for (int j = 0; j < layer->input_size; j++)
        {
            double mean_input_j = layer->mean_input->v[j];
            //Calculate Previous layer gradient error
            gradient_previous->v[j] += layer->weights[i].v[j] * gradient_i;
            //Update weights
            layer->weights[i].v[j] = optimizer->apply_gradient(layer->weights[i].v[j], gradient_i * mean_input_j, layer_index, i, optimizer);
        }
    }
    //Return gradient error tensor
    return gradient_previous;
}

void save_layer(FILE *fp, layer *layer)
{
    fprintf(fp, "input_size:%d, output_size:%d, type:%d\n", layer->input_size, layer->output_size, layer->type);
    for (int i = 0; i < layer->output_size; i++)
    {
        save_tensor(fp, &layer->weights[i]);
    }
    save_tensor(fp, &layer->biases);
    save_activation(fp, layer->activation);
}

layer *read_layer(FILE *fp)
{
    int type, input_size, output_size;
    fscanf(fp, "input_size:%d, output_size:%d, type:%d\n", &input_size, &output_size, &type);
    if (type >= 0)
    {
        layer *layer = build_layer(type, output_size, NULL);
        layer->input_size = input_size;
        layer->weights = malloc(sizeof(tensor) * layer->output_size);
        for (int i = 0; i < layer->output_size; i++)
        {
            read_tensor(fp, &layer->weights[i], layer->input_size);
        }
        read_tensor(fp, &layer->biases, layer->output_size);
        layer->activation = read_activation(fp);
        return layer;
    }
}