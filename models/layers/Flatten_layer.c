#include "../../include/layer.h"


//Compile Flatten layer
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

//Forward calculation function for Flatten layer when training
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
        i++;
    }
    free(iterator);
    return output;
}

//Forward calculation function for Flatten layer when predicting
tensor *forward_calculation_predict_Flatten(const tensor *input, tensor *output, layer *layer)
{
    forward_calculation_training_Flatten(input, output, NULL, layer);
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
        int* iterator = get_iterator(gradient_previous);
        int output_index=0;
        while(!gradient_previous->is_done(gradient_previous, iterator))
        {
            if(layer_index>0)
            {
                gradient_previous->set_value(gradient_previous, iterator, gradient->v[output_index]);
            }
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

//Configure Flatten layer methods
void configure_layer_Flatten(layer* layer)
{
    //Set used methods for the layer
    configure_default_layer(layer);
    layer->parameters = NULL;
    layer->init_training_memory = init_memory_training_no_activation;
    layer->clear_training_memory= clear_layer_training_memory_no_activation;
    layer->forward_propagation_training_loop= forward_propagation_training_loop_no_activation;
    layer->forward_calculation_predict=forward_calculation_predict_Flatten;
    layer->forward_calculation_training=forward_calculation_training_Flatten;
    layer->backward_propagation_loop=backward_propagation_loop_Flatten;
    layer->backward_calculation=backward_calculation_Flatten;
    layer->compile_layer=compile_layer_Flatten;
}

//Build Flatten layer
layer* build_layer_Flatten()
{
    layer *layer = (struct layer *)malloc(sizeof(struct layer));
    configure_layer_Flatten(layer);
    layer->type = FLATTEN;
    layer->activation = NULL;
    layer->output_shape = build_shape(OneD);
    return layer;
}