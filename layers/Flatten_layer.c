#include "../include/layer.h"


void clear_parameters_Flatten(layer* layer)
{
}

void compile_layer_Flatten(shape* input_shape, layer *layer)
{
    layer->input_shape = clone_shape(input_shape);
    int output_size = 1;
    for(int i=0;i<input_shape->dimension;i++)
    {
        output_size*=input_shape->sizes[i];
    }
    layer->output_shape->sizes[0]=output_size;
}

void build_shape_list_Flatten(layer* layer, shape_list* shape_list)
{
    shape_list->n_shapes=0;
    shape_list->shapes=NULL;
}

tensor *forward_calculation_training_Flatten(const tensor *input, tensor *output, tensor* activation_input, layer *layer)
{
    int output_size = layer->output_shape->sizes[0];
    int* iterator = get_iterator(input);
    int i=0;
    while(!input->is_done(input, iterator))
    {
        double v = input->get_value(input, iterator);
        iterator = input->get_next(input, iterator);
        output->v[i]=v;
        activation_input->v[i] = output->v[i];
        i++;
    }
    if (layer->activation)
    {
        //Execute activation function and return output tensor
        output = layer->activation->activation_forward(output, layer->activation);
    }
    return output;
}

//Forward propagation function for Flatten layer
tensor *forward_calculation_predict_Flatten(const tensor *input, tensor *output, layer *layer)
{
    int output_size = layer->output_shape->sizes[0];
    int* iterator = get_iterator(input);
    int i=0;
    while(!input->is_done(input, iterator))
    {
        double v = input->get_value(input, iterator);
        iterator = input->get_next(input, iterator);
        output->v[i]=v;
        i++;
    }
    free(iterator);
    if (layer->activation)
    {
        //Execute activation function and return output tensor
        output = layer->activation->activation_forward(output, layer->activation);
    }
    return output;
}

//backward propagation loop for Flatten layer
tensor *backward_propagation_loop_Flatten(tensor *gradients, optimizer *optimizer, struct layer *layer, int layer_index)
{
    int output_size = layer->output_shape->sizes[0];
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
        int* iterator = get_iterator(gradient_previous);
        int output_index=0;
        while(!gradient_previous->is_done(gradient_previous, iterator))
        {
            double g_p_v=gradient_previous->get_value(gradient_previous, iterator);
            gradient_previous->set_value(gradient_previous, iterator, g_p_v+gradient->v[output_index]);
            iterator = gradient_previous->get_next(gradient_previous,iterator);
            gradient->v[output_index]=0;
            output_index++;
        }
    }
    return layer->previous_gradients;
}

//Flatten layer doesn't calculate anything and the function is not called  in backward loop
void backward_calculation_Flatten(optimizer *optimizer, layer *layer, int layer_index)
{
}

void save_parameters_Flatten(FILE* fp, layer* layer)
{
}

void read_parameters_Flatten(FILE* fp, layer* layer)
{
}

void configure_layer_Flatten(layer* layer)
{
    //Set used methods for the layer
    configure_default_layer(layer);
    layer->clear_parameters=clear_parameters_Flatten;
    layer->clear_training_memory= clear_layer_training_memory;
    layer->clear_predict_memory=clear_layer_predict_memory;
    layer->forward_calculation_predict=forward_calculation_predict_Flatten;
    layer->forward_calculation_training=forward_calculation_training_Flatten;
    layer->backward_propagation_loop=backward_propagation_loop_Flatten;
    layer->backward_calculation=backward_calculation_Flatten;
    layer->build_shape_list = build_shape_list_Flatten;
    layer->compile_layer=compile_layer_Flatten;
    layer->init_training_memory=init_memory_training;
    layer->save_parameters=save_parameters_Flatten;
    layer->read_parameters=read_parameters_Flatten;
    layer->parameters = NULL;
}

layer* build_layer_Flatten(activation* activation)
{
    layer *layer = (struct layer *)malloc(sizeof(struct layer));
    configure_layer_Flatten(layer);
    layer->type = FLATTEN;
    layer->activation = activation;
    layer->output_shape = build_shape(OneD);
    return layer;
}