#include "include/layer.h"
#include "include/tensor.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

//Clear memory of temporary stored inputs and outputs
void clear_layer_backpropagation_memory(layer *layer, tensor* weights_gradient, tensor* biases_gradient, tensor* gradients)
{
    #pragma omp parallel for
    for (int i = 0; i < layer->batch_size; i++)
    {
        clear_tensor(&layer->outputs[i]);
        clear_tensor(&layer->activation_input[i]);
        clear_tensor(&gradients[i]);
    }
    clear_tensors(weights_gradient, layer->output_size);
    free(gradients);
    clear_tensor(biases_gradient);
    free(layer->activation_input);
    free(layer->layer_input);
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
    layer->backward_propagation_loop = backward_propagation_loop;
    layer->forward_calculation = forward_calculation_FC;
    layer->backward_calculation = backward_calculation_FC;
}

//Common forward propagation loop
tensor *forward_propagation_loop(const tensor *inputs, int batch_size, short is_training, struct layer *layer, progression* progression)
{
    int output_size = layer->output_size;
    layer->is_training = is_training;
    if (is_training)
    {
        //Store inputs (used in backward propagation)
        layer->activation_input = (tensor *)malloc(sizeof(tensor)*batch_size);
        layer->layer_input = (tensor *)malloc(sizeof(tensor)*batch_size);
        layer->batch_size = batch_size;
    }
    tensor* activation_input;
    //Allocate memory for outputs and activation_input
    layer->outputs = malloc(sizeof(tensor) * batch_size);
    // Loop into input batch
    for (int i = 0; i < batch_size; i++)
    {
        //Output tensor memory allocation
        tensor *output = &layer->outputs[i];
        initialize_tensor(output, output_size);
        if(layer->is_training)
        {
            activation_input = &layer->activation_input[i];
            initialize_tensor(activation_input, output_size);
        }
        if (is_training)
        {
            layer->layer_input[i]=inputs[i];
        }
        //Execute specific forward propagation
        layer->forward_calculation(&inputs[i], output, activation_input, layer);
        if(progression)
        {
            progression->call_back(progression);
        }
    }
    return layer->outputs;
}

//Common backward propagation loop
tensor *backward_propagation_loop(tensor *gradients, optimizer *optimizer, struct layer *layer, int layer_index)
{
    int output_size = layer->output_size;
    int input_size = layer->input_size;
    int batch_size = layer->batch_size;
    //Previous layer gradient error tensor memory allocation
    tensor *gradient_previous_batch = (tensor *)malloc(sizeof(tensor)*batch_size);
    //Biases used to update layer biases
    tensor biases_gradient;
    //Weights used to update layer weights
    tensor* weights_gradient =(tensor*)malloc(sizeof(tensor)*output_size);
    for(int i=0;i<output_size;i++)
    {
        initialize_tensor(&weights_gradient[i], input_size);
    }
    initialize_tensor(&biases_gradient, output_size);
    for(int i=0;i<batch_size;i++)
    {
        tensor* gradient = &gradients[i];
        tensor* gradient_previous = &gradient_previous_batch[i];
        initialize_tensor(gradient_previous, input_size);
        if (layer->activation)
        {
            //Back propagate the gradient error tensor
            gradient = layer->activation->activation_backward_propagation(&layer->activation_input[i], gradient, &layer->outputs[i], layer->activation);
        }
        //biases_gradient is the sum of batch gradients
        #pragma omp parallel for
        for(int j=0;j<output_size;j++)
        {
            double gradient_j = gradient->v[j];
            biases_gradient.v[j]+=gradient_j;
            for (int k = 0; k < input_size; k++)
            {
                double input_k = layer->layer_input[i].v[k];
                //Calculate Previous layer gradient error
                gradient_previous->v[k] += layer->weights[j].v[k] * gradient_j;
                //Gradient weights is the sum of batch gradient multiplied by batch input;
                weights_gradient[j].v[k] += input_k*gradient_j;
            }
        }
    }
    layer->backward_calculation(&biases_gradient, weights_gradient, optimizer, layer, layer_index);
    clear_layer_backpropagation_memory(layer, weights_gradient, &biases_gradient, gradients);
    return gradient_previous_batch;
}

//Forward propagation function for Fully Connected layer (perceptron)
tensor *forward_calculation_FC(const tensor *input, tensor *output, tensor* activation_input, layer *layer)
{
    //Loop into output tensor
    #pragma omp parallel for
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
            activation_input->v[i] = output->v[i];
        }
    }
    if (layer->activation)
    {
        //Execute activation function and return output tensor
        output = layer->activation->activation_forward(output, layer->activation);
    }
}

//Backward propagation function for Fully Connected layer (perceptron)
void backward_calculation_FC(tensor *biases_gradient, tensor *weights_gradient, optimizer *optimizer, layer *layer, int layer_index)
{
    int output_size = layer->output_size;
    int input_size = layer->input_size;
    //Update biases using new gradient
    #pragma omp parallel for
    for (int i = 0; i < output_size; i++)
    {
        //Optimizer update bias
        layer->biases.v[i] = optimizer->apply_gradient(layer->biases.v[i], biases_gradient->v[i], layer_index, i, optimizer);
        //Calculate the gradient for previous layer and update weights
        for (int j = 0; j < input_size; j++)
        {
            //Update weights
            layer->weights[i].v[j] = optimizer->apply_gradient(layer->weights[i].v[j], weights_gradient[i].v[j], layer_index, i, optimizer);
        }
    }
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