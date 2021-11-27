#include "include/layer.h"
#include "include/tensor.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

//Clear memory of temporary stored inputs and outputs
void clear_layer_training_memory_FC(layer *layer)
{
    #pragma omp parallel for
    for (int i = 0; i < layer->batch_size; i++)
    {
        clear_tensor(&layer->outputs[i]);
        clear_tensor(&layer->activation_input[i]);
        clear_tensor(&layer->previous_gradients[i]);
    }
    clear_tensors(layer->weights_gradients, layer->output_size);
    free(layer->previous_gradients);
    clear_tensor(&layer->biases_gradients);
    free(layer->activation_input);
    free(layer->layer_inputs);
    free(layer->outputs);
    free(layer->weights_gradients);
}

void clear_layer_predict_memory_FC(layer* layer)
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

void init_memory_training_FC(layer* layer)
{
    int input_size = layer->input_size;
    int output_size = layer->output_size;
    int batch_size = layer->batch_size;
    layer->layer_inputs=(tensor*) malloc(sizeof(tensor)*batch_size);
    layer->activation_input = (tensor *)malloc(sizeof(tensor)*batch_size);
    layer->previous_gradients = (tensor *)malloc(sizeof(tensor)*batch_size);
    layer->weights_gradients =(tensor*)malloc(sizeof(tensor)*output_size);
    initialize_tensor(layer->previous_gradients, input_size);
    initialize_tensor(&layer->biases_gradients, output_size);
    for(int i=0;i<output_size;i++)
    {
        initialize_tensor(&layer->weights_gradients[i], input_size);
    }
    layer->outputs = malloc(sizeof(tensor) * batch_size);
    #pragma omp parallel
    for(int i=0;i<batch_size;i++)
    {
        initialize_tensor(&layer->outputs[i], output_size);
        initialize_tensor(&layer->activation_input[i], output_size);
        initialize_tensor(&layer->previous_gradients[i], input_size);
    }
}

void init_memory_predict_FC(layer* layer)
{
    int batch_size = layer->batch_size;
    int output_size = layer->output_size;
    layer->outputs = malloc(sizeof(tensor) * batch_size);
    for(int i=0;i<batch_size;i++)
    {
        initialize_tensor(&layer->outputs[i], output_size);
    }
}

//Common forward propagation loop
tensor *forward_propagation_training_loop(const tensor *inputs, int batch_size, struct layer *layer, progression* progression)
{
    int output_size = layer->output_size;
    int input_size = layer->input_size;
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
        if(progression)
        {
            progression->call_back(progression);
        }
    }
    return layer->outputs;
}

tensor *forward_propagation_predict_loop(const tensor *inputs, int batch_size, struct layer *layer, progression* progression)
{
    int output_size = layer->output_size;
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

//Common backward propagation loop
tensor *backward_propagation_loop(tensor *gradients, optimizer *optimizer, struct layer *layer, int layer_index)
{
    int output_size = layer->output_size;
    int input_size = layer->input_size;
    int batch_size = layer->batch_size;
    #pragma omp parallel for
    for(int i=0;i<batch_size;i++)
    {
        tensor* gradient = &gradients[i];
        tensor* gradient_previous = &layer->previous_gradients[i];
        tensor* output = &layer->outputs[i];
        if (layer->activation)
        {
            //Back propagate the gradient error tensor
            gradient = layer->activation->activation_backward_propagation(&layer->activation_input[i], gradient, output, layer->activation);
        }
        //biases_gradient is the sum of batch gradients
        for(int j=0;j<output_size;j++)
        {
            output->v[j]=0;
            double gradient_j = gradient->v[j];
            gradient->v[j]=0;
            layer->biases_gradients.v[j]+=gradient_j;
            for (int k = 0; k < input_size; k++)
            {
                double input_k = layer->layer_inputs[i].v[k];
                if(layer_index>0)
                {
                    //Calculate Previous layer gradient error
                    gradient_previous->v[k] += layer->weights[j].v[k] * gradient_j;
                }
                //Gradient weights is the sum of batch gradient multiplied by batch input;
                layer->weights_gradients[j].v[k] += input_k*gradient_j;
            }
        }
    }
    layer->backward_calculation(&layer->biases_gradients, layer->weights_gradients, optimizer, layer, layer_index);
    return layer->previous_gradients;
}

tensor *forward_calculation_training_FC(const tensor *input, tensor *output, tensor* activation_input, layer *layer)
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
        //Store the activation input
        activation_input->v[i] = output->v[i];
    }
    if (layer->activation)
    {
        //Execute activation function and return output tensor
        output = layer->activation->activation_forward(output, layer->activation);
    }
}

//Forward propagation function for Fully Connected layer (perceptron)
tensor *forward_calculation_predict_FC(const tensor *input, tensor *output, layer *layer)
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
    for (int i = 0; i < output_size; i++)
    {
        //Optimizer update bias
        layer->biases.v[i] = optimizer->apply_gradient(layer->biases.v[i], biases_gradient->v[i], layer_index, i, optimizer);
        biases_gradient->v[i]=0;
        //Calculate the gradient for previous layer and update weights
        for (int j = 0; j < input_size; j++)
        {
            //Update weights
            layer->weights[i].v[j] = optimizer->apply_gradient(layer->weights[i].v[j], weights_gradient[i].v[j], layer_index, i, optimizer);
            weights_gradient[i].v[j]=0;
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

void build_layer_FC(layer *layer)
{

    //Set used methods for the layer
    layer->compile_layer = compile_layer;
    layer->init_predict_memory = init_memory_predict_FC;
    layer->init_training_memory = init_memory_training_FC;
    layer->clear_predict_memory = clear_layer_predict_memory_FC;
    layer->clear_training_memory = clear_layer_training_memory_FC;
    layer->forward_calculation_training = forward_calculation_training_FC;
    layer->forward_calculation_predict = forward_calculation_predict_FC;
    layer->backward_calculation = backward_calculation_FC;
}

layer *build_layer(layer_type type, int output_size, activation *activation)
{
    layer *layer = (struct layer *)malloc(sizeof(struct layer));
    layer->type = type;
    //Store input and output size
    layer->output_size = output_size;
    //Store activation function
    layer->activation = activation;
    layer->forward_propagation_training_loop = forward_propagation_training_loop;
    layer->forward_propagation_predict_loop = forward_propagation_predict_loop;
    layer->backward_propagation_loop = backward_propagation_loop;
    switch (type)
    {
    default:
        build_layer_FC(layer);
        break;
    }
    return layer;
}