#include "include/layer.h"
#include "include/tensor.h"
#include <stdlib.h>

//Clear memory of temporary stored inputs and outputs
void clear_layer_input_output(layer* layer)
{
    clear_tensor(layer->sum_input);
    clear_tensor(layer->sum_activation_input);
    clear_tensor(layer->sum_output);
    for(int i=0;i<layer->batch_size;i++)
    {
        clear_tensor(&layer->outputs[i]);
    }
    free(layer->sum_activation_input);
    free(layer->sum_input);
    free(layer->sum_output);
    free(layer->outputs);
}

void clear_layer(layer* layer)
{
    clear_tensors(layer->weights, layer->output_size);
    free(layer->weights);
    clear_tensor(&layer->biases);
    if(layer->activation)
    {
        free(layer->activation);
    }
}

void compile_layer(int input_size, layer* layer)
{
    layer->input_size = input_size;
    layer->weights =(tensor*)malloc(layer->output_size* sizeof(tensor));
    double invert_rand_max = (double)1.0/(double)RAND_MAX;
    for(int i=0;i<layer->output_size;i++)
    {
        initialize_tensor(&layer->weights[i], input_size);
        for(int j=0;j<input_size;j++)
        {
            layer->weights[i].v[j] = ((double)rand() * invert_rand_max) -(double)0.5 ;
        }      
    }
    //Initialize biases
    initialize_tensor(&layer->biases, layer->output_size);
    for(int i=0;i<layer->output_size;i++)
    {
        layer->biases.v[i]=((double)rand() * invert_rand_max) -(double)0.5 ;
    }
}

layer* build_layer_FC(int output_size, activation* activation){
    //Allocate layer memory
    layer* result = (layer*)malloc(sizeof(layer));
    result->type = FC;
    //Store input and output size
    result->output_size = output_size;
    
    //Set used methods for the layer
    result->compile_layer = compile_layer;
    result->forward_propagation_loop=forward_propagation_loop;
    result->backward_propagation = backward_propagation;
    result->forward_calculation=forward_calculation_FC;
    result->backward_calculation=backward_calculation_FC;
    //Store activation function
    result->activation = activation;
    //Return configured and initialized FC layer
    return result;
}

//Common forward propagation loop
tensor* forward_propagation_loop(tensor* inputs, int batch_size, short is_training, struct layer* layer)
{
    int output_size=layer->output_size;
    layer->is_training = is_training;
    if(is_training)
    {
        layer->sum_activation_input = (tensor*)malloc(sizeof(tensor));
        layer->sum_output = (tensor*)malloc(sizeof(tensor));
        layer->invert_output_size = (double)1.0/output_size;
        layer->sum_input = (tensor*)malloc(sizeof(tensor));
        initialize_tensor(layer->sum_activation_input, output_size);
        initialize_tensor(layer->sum_output, output_size);
        initialize_tensor(layer->sum_input, layer->input_size);
        //Store inputs (used in backward propagation)
        layer->batch_size = batch_size;
    }

    //Allocate memory for outputs and activation_input
    layer->outputs=malloc(sizeof(tensor)*batch_size);
    // Loop into input batch
    for(int i=0;i<batch_size;i++)
    {
        //Output tensor memory allocation
        tensor* output = &layer->outputs[i];
        initialize_tensor(output, output_size);
        if(is_training)
        {
            for(int j=0;j<layer->input_size;j++)
            {
                layer->sum_input->v[j]+=(inputs[i].v[j]);
            }
        }
        //Execute specific forward propagation
        layer->forward_calculation(&inputs[i], output, layer);
    }
    return layer->outputs; 
}

//Common backward propagation loop
tensor* backward_propagation(tensor* mean_gradient, optimizer* optimizer, struct layer* layer, int layer_index)
{
    //Previous layer gradient error tensor memory allocation
    tensor* gradient_previous = (tensor*)malloc(sizeof(tensor));
    initialize_tensor(gradient_previous, layer->input_size);
    layer->backward_calculation(mean_gradient, gradient_previous, optimizer, layer, layer_index);
    if(layer->is_training)
    {
        clear_tensor(mean_gradient);
        free(mean_gradient);
        clear_layer_input_output(layer);
    }
    return gradient_previous;
}

//Forward propagation function for Fully Connected layer (perceptron)
tensor* forward_calculation_FC(tensor* input, tensor* output,layer* layer){
    //Loop into output tensor
    for(int j=0;j<layer->output_size;j++)
    {
            //Loop into input tensor
            for(int k=0;k<layer->input_size;k++){
                //sum weighted input element using weights matrix
                output->v[j] += layer->weights[j].v[k]* (input->v[k]);
            }
            //Add bias
            output->v[j] += layer->biases.v[j];
            if(layer->is_training)
            {
                //Store the activation input
                layer->sum_activation_input->v[j] += (output->v[j]); 
            }           
    }
    if(layer->activation)
    {
        //Execute activation function and return output tensor
        output = layer->activation->activation_func(output);
    }
    if(layer->is_training)
    {
        for(int j=0;j<layer->output_size;j++)
        {
            layer->sum_output->v[j]+=(output->v[j]);           
        }
    }
}

//Backward propagation function for Fully Connected layer (perceptron)
tensor* backward_calculation_FC(tensor* gradient, tensor* gradient_previous, optimizer* optimizer, layer* layer, int layer_index)
{
    if(layer->activation)
    {
        //Back propagate the gradient error tensor
        gradient = layer->activation->activation_backward_propagation(layer->sum_activation_input,gradient, layer->sum_output, layer->activation);
    }
    //Update biases using new gradient
    for(int j=0;j<layer->output_size;j++)
    {       
        //Optimizer update bias using the activation primitive
        layer->biases.v[j]=optimizer->apply_gradient(layer->biases.v[j], gradient->v[j],layer_index, j, optimizer);           
    }
    //Calculate the gradient for previous layer and update weights
    for(int j=0;j<layer->input_size;j++){
        double mean_input_j = layer->sum_input->v[j];
        for(int k=0;k<layer->output_size;k++){
            //Calculate Previous layer gradient error
            gradient_previous->v[j] += layer->weights[k].v[j]*(gradient->v[k]);
            //Update weights
            layer->weights[k].v[j]=optimizer->apply_gradient(layer->weights[k].v[j], gradient->v[k]*mean_input_j, layer_index, k, optimizer);
        }
    }
    //Return gradient error tensor
    return gradient_previous;
}