#include "include/layer.h"
#include "include/tensor.h"
#include <stdlib.h>

//Clear memory of temporary stored inputs and outputs
void clear_layer_input_output(layer* layer){
    for(int i=0;i<layer->batch_size;i++)
    {
        clear_tensor(&layer->outputs[i]);
    }
    clear_tensor(layer->mean_activation_input);
    clear_tensor(layer->mean_input);
    clear_tensor(layer->mean_output);
    free(layer->mean_activation_input);
    free(layer->mean_input);
    free(layer->mean_output);
    free(layer->outputs);
}

void clear_layer(layer* layer)
{
    clear_layer_input_output(layer);
    clear_tensor(&layer->weights);
    clear_tensor(&layer->biases);
    free(layer->activation);
    free(layer);
}

layer* build_layer_FC(int input_size, int output_size, activation* activation){
    //Allocate layer memory
    layer* result = (layer*)malloc(sizeof(layer));
    //Store input and output size
    result->input_size = input_size;
    result->output_size = output_size;
    //Initialize weights matrix which is input_sizeXoutput_size
    result->weights.size = input_size*output_size;
    for(int i=0;i<input_size*output_size;i++){
        result->weights.v[i] = (double)rand() / (double)RAND_MAX ;
    }
    //Initialize biases
    result->biases.size = output_size;
    result->biases.v = calloc(output_size,sizeof(double));
    //Set used methods for the layer
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
    layer->is_training = is_training;
    if(is_training)
    {
        layer->mean_activation_input->size = layer->output_size;
        layer->mean_activation_input->v = calloc(layer->output_size,sizeof(double));
        layer->mean_output->size = layer->output_size;
        layer->mean_output->v = calloc(layer->output_size,sizeof(double));
        layer->mean_input->size = layer->input_size;
        layer->mean_input->v = calloc(layer->input_size,sizeof(double));
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
        output->size = layer->output_size;
        output->v = calloc(layer->output_size,sizeof(double));
        for(int j=0;j<layer->input_size;j++)
        {
            layer->mean_input->v[j]+=(inputs[i].v[j]/batch_size);
        }
        //Execute specific forward propagation
        layer->forward_calculation(&inputs[i], output, layer);
    }
    return layer->outputs; 
}

//Common backward propagation loop
tensor* backward_propagation(tensor* mean_gradient, optimizer* optimizer, struct layer* layer)
{
    //Previous layer gradient error tensor memory allocation
    tensor* gradient_previous = (tensor*)malloc(sizeof(tensor));
    layer->backward_calculation(mean_gradient, gradient_previous, optimizer, layer);
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
                output->v[j] += layer->weights.v[(j*layer->input_size)+k]* (input->v[k]);
            }
            //Add bias
            output->v[j] += layer->biases.v[j];
            if(layer->is_training)
            {
                //Store the activation input
                layer->mean_activation_input->v[j] += (output->v[j]/layer->batch_size); 
            }           
    }
    //Execute activation function and return output tensor
    output = layer->activation->activation_func(output);
    if(layer->is_training)
    {
        for(int j=0;j<layer->output_size;j++)
        {
            layer->mean_output->v[j]+=(output->v[j]/layer->batch_size);           
        }
    }
}

//Backward propagation function for Fully Connected layer (perceptron)
tensor* backward_calculation_FC(tensor* gradient, tensor* gradient_previous, optimizer* optimizer, layer* layer)
{
    //Back propagate the gradient error tensor
    gradient = layer->activation->activation_backward_propagation(layer->mean_activation_input,gradient, layer->mean_output, layer->activation);
    //Update biases using new gradient
    for(int j=0;j<layer->output_size;j++)
    {       
        //Optimizer update bias using the activation primitive
        layer->biases.v[j]=optimizer->apply_gradient(layer->biases.v[j], gradient->v[j], optimizer);           
    }
    //Calculate the gradient for previous layer and update weights
    for(int j=0;j<layer->input_size;j++){
        for(int k=0;k<layer->output_size;k++){
            //Calculate Previous layer gradient error
            gradient_previous->v[j] += layer->weights.v[j+(layer->input_size*k)]*(gradient->v[k]);
            //Update weights
            layer->weights.v[k*layer->input_size+j]=optimizer->apply_gradient(layer->weights.v[k*layer->input_size+j], gradient->v[k]*layer->mean_input->v[j], optimizer);
        }
    }
    //Return gradient error tensor
    return gradient_previous;
}