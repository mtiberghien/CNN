#include "include/layer.h"
#include "include/tensor.h"
#include <stdlib.h>

//Clear memory of temporary stored inputs and outputs
void clear_layer_input_output(layer* layer){
    for(int i=0;i<layer->batch_size;i++)
    {
        clear_tensor(layer->inputs[i]);
        clear_tensor(layer->outputs[i]);
        clear_tensor(layer->activation_inputs[i]);
    }
    free(layer->inputs);
    free(layer->outputs);
    free(layer->activation_inputs);
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
    result->backward_propagation_loop = backward_propagation_loop;
    result->forward_propagation=forward_propagation_FC;
    result->backward_propagation=backward_propagation_FC;
    //Store activation function
    result->activation = activation;
    //Return configured and initialized FC layer
    return result;
}

//Common forward propagation loop
tensor* forward_propagation_loop(tensor* inputs, int n_inputs, struct layer* layer)
{
    // Clear memory
    clear_layer_input_output(layer);
    //Store inputs (used in backward propagation)
    layer->inputs = inputs;
    layer->batch_size = n_inputs;
    //Allocate memory for outputs and activation_input
    layer->outputs=malloc(sizeof(tensor)*n_inputs);
    layer->activation_inputs=malloc(sizeof(tensor)*n_inputs);
    // Loop into input batch
    for(int i=0;i<n_inputs;i++)
    {
        //Output tensor memory allocation
        tensor output = layer->outputs[i];
        output.size = layer->output_size;
        output.v = calloc(layer->output_size,sizeof(double));
        //Execute specific forward propagation
        layer->forward_propagation(&inputs[i], &output, &layer->activation_inputs[i], layer);
    }
    return layer->outputs; 
}

//Common backward propagation loop
tensor* backward_propagation_loop(tensor* output_errors, optimizer* optimizer, struct layer* layer)
{
    //Previous layer gradient error tensor memory allocation
    tensor* input_error = (tensor*)malloc(sizeof(tensor)*layer->batch_size);
    //Loop into following layer gradient error batch
    for(int i=0;i<layer->batch_size;i++)
    {
        //Output gradient error tensor
        tensor output_error = output_errors[i];
        //Activation Output gradient error memory allocation
        tensor aoe_t;
        aoe_t.size = layer->output_size;
        aoe_t.v = (double*)malloc(sizeof(double)*aoe_t.size);
        //Previous layer gradient error tensor memory allocation
        tensor ie_t = input_error[i];
        ie_t.size = layer->input_size;
        ie_t.v = calloc(layer->input_size, sizeof(double));
        //Execute specific backward propagation
        layer->backward_propagation(&output_error,&aoe_t, &ie_t, &layer->inputs[i], optimizer, layer);
        //Clear gradient error from next layer
        clear_tensor(aoe_t);
        clear_tensor(output_error);
    }
    free(output_errors);
    return input_error;
}

//Forward propagation function for Fully Connected layer (perceptron)
tensor* forward_propagation_FC(tensor* input, tensor* output, tensor* activation_input, layer* layer){
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
            //Store the activation input
            activation_input->v[j] = output->v[j];
            //Execute activation function
            output->v[j] = layer->activation->activation_func(output->v[j]);
    }
    //Return output tensor
    return output;
}

//Backward propagation function for Fully Connected layer (perceptron)
tensor* backward_propagation_FC(tensor* output_error, tensor* activation_output_error, tensor* input_error, tensor* input, optimizer* optimizer, layer* layer)
{
    //Calculate the gradient for current layer activation and update biases using these gradient
    for(int j=0;j<layer->output_size;j++)
    {
        //Get the primitive from gradient error of next layer
        activation_output_error->v[j]=layer->activation->activation_func_prime(output_error->v[j]);
        //Optimizer update bias using the activation primitive
        layer->biases.v[j]=optimizer->apply_gradient(layer->biases.v[j], activation_output_error->v[j], optimizer);           
    }
    //Calculate the gradient for previous layer and update weights
    for(int j=0;j<layer->input_size;j++){
        for(int k=0;k<layer->output_size;k++){
            //Calculate Previous layer gradient error
            input_error->v[j] += layer->weights.v[j+(layer->input_size*k)]*(activation_output_error->v[k]);
            //Update weights
            layer->weights.v[k*layer->input_size+j]=optimizer->apply_gradient(layer->weights.v[k*layer->input_size+j], activation_output_error->v[k]*input->v[j], optimizer);
        }
    }
    //Return gradient error tensor
    return input_error;
}