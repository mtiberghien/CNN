#include "../include/layer.h"
#include "../include/tensor.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

typedef struct fc_parameters{
    //Stores the weight matrix as an array of tensor
    tensor weights;
    tensor weights_gradients;
    //Stores the biases
    tensor biases;
    tensor biases_gradients;
}fc_parameters;

//Clear memory of temporary stored inputs and outputs
void clear_layer_training_memory_FC(layer *layer)
{
    fc_parameters* params = (fc_parameters*)layer->parameters;
    clear_layer_training_memory(layer);
    clear_tensor(&params->weights_gradients);
    clear_tensor(&params->biases_gradients);
}

void clear_parameters_FC(layer* layer)
{
    fc_parameters* params = (fc_parameters*)layer->parameters;
    clear_tensor(&params->weights);
    clear_tensor(&params->biases);
}

void compile_layer_FC(shape* input_shape, layer *layer)
{
    fc_parameters* params = (fc_parameters*)layer->parameters;
    layer->input_shape = clone_shape(input_shape);
    shape* weights_shape = build_shape(TwoD);
    weights_shape->sizes[0]=layer->output_shape->sizes[0];
    weights_shape->sizes[1]=input_shape->sizes[0];
    initialize_tensor(&params->weights, weights_shape);
    double invert_rand_max = (double)1.0 / (double)RAND_MAX;
    double limit = sqrt((double)6 / (weights_shape->sizes[0] + weights_shape->sizes[1]));
    clear_shape(weights_shape);
    free(weights_shape);
    int* iterator = get_iterator(&params->weights);
    while(!params->weights.is_done(&params->weights, iterator))
    {
        params->weights.set_value(&params->weights, iterator, (2 * limit * ((double)rand() * invert_rand_max)) - limit);
        iterator = params->weights.get_next(&params->weights, iterator);
    }
    free(iterator);
    //Initialize biases
    initialize_tensor(&params->biases, layer->output_shape);
    for (int i = 0; i < layer->output_shape->sizes[0]; i++)
    {
        params->biases.v[i] = (2 * limit * ((double)rand() * invert_rand_max)) - limit;
    }
}

void init_memory_training_FC(layer* layer)
{
    init_memory_training(layer);
    fc_parameters* params = (fc_parameters*)layer->parameters;
    initialize_tensor(&params->weights_gradients, params->weights.shape);
    initialize_tensor(&params->biases_gradients, layer->output_shape);
}

void build_shape_list_FC(layer* layer, shape_list* shape_list)
{
    fc_parameters* params = (fc_parameters*)layer->parameters;
    shape_list->n_shapes=2;
    shape_list->shapes=malloc(sizeof(shape)*2);
    shape_list->shapes[0]=*clone_shape(params->biases.shape);
    shape_list->shapes[1]=*clone_shape(layer->output_shape);
}

tensor *forward_calculation_training_FC(const tensor *input, tensor *output, tensor* activation_input, layer *layer)
{
    fc_parameters* params = (fc_parameters*)layer->parameters;
    int output_size = layer->output_shape->sizes[0];
    int input_size = layer->input_shape->sizes[0];
    double** weights = (double**)params->weights.v;
    //Loop into output tensor
    for (int i = 0; i < output_size; i++)
    {
        //Loop into input tensor
        for (int j = 0; j < input_size; j++)
        {
            //sum weighted input element using weights matrix
            output->v[i] += weights[i][j] * (input->v[j]);
        }
        //Add bias
        output->v[i] += params->biases.v[i];
        //Store the activation input (used in backpropagation)
        activation_input->v[i] = output->v[i];
    }
    if (layer->activation)
    {
        //Execute activation function and return output tensor
        output = layer->activation->activation_forward(output, layer->activation);
    }
    return output;
}

//Forward propagation function for Fully Connected layer (perceptron)
tensor *forward_calculation_predict_FC(const tensor *input, tensor *output, layer *layer)
{
    fc_parameters* params = (fc_parameters*)layer->parameters;
    int output_size = layer->output_shape->sizes[0];
    int input_size = layer->input_shape->sizes[0];
    double** weights = (double**)params->weights.v;
    //Loop into output tensor
    for (int i = 0; i < output_size; i++)
    {
        //Loop into input tensor
        for (int j = 0; j < input_size; j++)
        {
            //sum weighted input element using weights matrix
            output->v[i] += weights[i][j] * (input->v[j]);
        }
        //Add bias
        output->v[i] += params->biases.v[i];
    }
    if (layer->activation)
    {
        //Execute activation function and return output tensor
        output = layer->activation->activation_forward(output, layer->activation);
    }
    return output;
}

//backward propagation loop for FC layers
tensor *backward_propagation_loop_FC(tensor *gradients, optimizer *optimizer, struct layer *layer, int layer_index)
{
    fc_parameters* params = (fc_parameters*)layer->parameters;
    int output_size = layer->output_shape->sizes[0];
    int input_size = layer->input_shape->sizes[0];
    int batch_size = layer->batch_size;
    double** weights = (double**)params->weights.v;
    double** weights_gradients = (double**) params->weights_gradients.v;
    double* biases_gradients= (double*)params->biases_gradients.v;
    #pragma omp parallel for
    for(int i=0;i<batch_size;i++)
    {
        tensor* gradient = &gradients[i];
        double* gradient_previous = layer->previous_gradients[i].v;
        tensor* output = &layer->outputs[i];
        if (layer->activation)
        {
            //Back propagate the gradient error tensor
            gradient = layer->activation->activation_backward_propagation(&layer->activation_input[i], gradient, output, layer->activation);
        }
        for(int j=0;j<output_size;j++)
        {
            output->v[j]=0;
            double gradient_j = gradient->v[j];
            gradient->v[j]=0;
            //biases_gradient is the sum of batch gradients
            biases_gradients[j]+=gradient_j;
            for (int k = 0; k < input_size; k++)
            {
                double input_k = layer->layer_inputs[i].v[k];
                if(layer_index>0)
                {
                    //Calculate Previous layer gradient error
                    gradient_previous[k] += weights[j][k] * gradient_j;
                }
                //Gradient weights is the sum of batch gradient multiplied by batch input;
                weights_gradients[j][k] += input_k*gradient_j;
            }
        }
    }
    layer->backward_calculation(optimizer, layer, layer_index);
    return layer->previous_gradients;
}

//Backward propagation function for Fully Connected layer (perceptron)
void backward_calculation_FC(optimizer *optimizer, layer *layer, int layer_index)
{
    fc_parameters* params = (fc_parameters*)layer->parameters;
    int output_size = layer->output_shape->sizes[0];
    int input_size = layer->input_shape->sizes[0];
    double** weights = (double**)params->weights.v;
    double** weights_gradient = (double**)params->weights_gradients.v;
    double* biases_gradient = params->biases_gradients.v;
    //Update biases using new gradient
    for (int i = 0; i < output_size; i++)
    {
        //Optimizer update bias
        params->biases.v[i] = optimizer->apply_gradient(params->biases.v[i], biases_gradient[i], layer_index, 0, &i, optimizer);
        biases_gradient[i]=0;
        //Calculate the gradient for previous layer and update weights
        for (int j = 0; j < input_size; j++)
        {
            //Update weights
            weights[i][j] = optimizer->apply_gradient(weights[i][j], weights_gradient[i][j], layer_index,1, &i, optimizer);
            weights_gradient[i][j]=0;
        }
    }
}

void save_trainable_parameters_FC(FILE* fp, layer* layer)
{
    fc_parameters* params = (fc_parameters*)layer->parameters;
    save_tensor(fp, &params->weights);
    save_tensor(fp, &params->biases);
}

void read_trainable_parameters_FC(FILE* fp, layer* layer)
{
    fc_parameters* params = (fc_parameters*)layer->parameters;
    read_tensor(fp, &params->weights);
    read_tensor(fp, &params->biases);
}

int get_trainable_parameters_count_FC(layer* layer)
{
    int total = layer->output_shape->sizes[0];
    total+=total*layer->input_shape->sizes[0];
    return total;
}

void configure_layer_FC(layer* layer)
{
    //Set used methods for the layer
    configure_default_layer(layer);
    fc_parameters* params = malloc(sizeof(fc_parameters));
    layer->parameters = params;
    layer->compile_layer = compile_layer_FC;
    layer->init_training_memory = init_memory_training_FC;
    layer->clear_training_memory = clear_layer_training_memory_FC;
    layer->forward_calculation_training = forward_calculation_training_FC;
    layer->forward_calculation_predict = forward_calculation_predict_FC;
    layer->backward_calculation = backward_calculation_FC;
    layer->backward_propagation_loop = backward_propagation_loop_FC;
    layer->build_shape_list = build_shape_list_FC;
    layer->clear_parameters = clear_parameters_FC;
    layer->read_trainable_parameters = read_trainable_parameters_FC;
    layer->save_trainable_parameters = save_trainable_parameters_FC;
    layer->get_trainable_parameters_count = get_trainable_parameters_count_FC;
}

layer* build_layer_FC(int output_size, activation* activation)
{
    layer *layer = (struct layer *)malloc(sizeof(struct layer));
    configure_layer_FC(layer);
    layer->type = FC;
    layer->output_shape = build_shape(OneD);
    layer->output_shape->sizes[0]=output_size;
    layer->activation = activation;
    return layer;
}