#include "../include/layer.h"

typedef struct padding2D_parameters{
    int padding_height;
    int padding_width;
} padding2D_parameters;


void save_parameters_Padding2D(FILE* fp, layer* layer)
{
    padding2D_parameters* params = (padding2D_parameters*)layer->parameters;
    fprintf(fp, "padding_width:%d, padding_height:%d\n", params->padding_width, params->padding_height);
}

void read_parameters_Padding2D(FILE* fp, layer* layer)
{
    padding2D_parameters* params = (padding2D_parameters*)layer->parameters;
    fscanf(fp, "padding_width:%d, padding_height:%d\n", &params->padding_width, &params->padding_height);
}

//Forward calculation function for Flatten layer when training
tensor *forward_calculation_training_Padding2D(const tensor *input, tensor *output, tensor* activation_input, layer *layer)
{
    padding2D_parameters* params = (padding2D_parameters*)layer->parameters;
    int input_height = input->shape->sizes[1];
    int input_width = input->shape->sizes[2];
    int input_channels = input->shape->sizes[0];
    int offset_x = params->padding_width/2;
    int offset_y = params->padding_height/2;
    double*** out = (double***)output->v;
    double*** in = (double***)input->v;
    for(int i=0;i<input_channels;i++)
    {
        double** matrix_out = out[i];
        double** matrix_in = in[i];
        for(int j=offset_y;j<input_height;j++)
        {
            for(int k=offset_x;k<input_width;k++)
            {
                matrix_out[j][k]=matrix_in[j][k];
            }
        }
    }
    return output;
}

//Forward calculation function for Flatten layer when predicting
tensor *forward_calculation_predict_Padding2D(const tensor *input, tensor *output, layer *layer)
{
    forward_calculation_training_Padding2D(input, output, NULL, layer);
}

//backward propagation loop for Flatten layer
tensor *backward_propagation_loop_Padding2D(tensor *gradients, optimizer *optimizer, struct layer *layer, int layer_index)
{
    int output_size = layer->output_shape->sizes[0];
    int batch_size = layer->batch_size;
    if(layer_index>0)
    {
        #pragma omp parallel for
        for(int i=0;i<batch_size;i++)
        {
            tensor* gradient = &gradients[i];
            tensor* previous_gradient = &layer->previous_gradients[i];
            padding2D_parameters* params = (padding2D_parameters*)layer->parameters;
            int output_height = previous_gradient->shape->sizes[1];
            int output_width = previous_gradient->shape->sizes[2];
            int output_channels = previous_gradient->shape->sizes[0];
            int offset_x = params->padding_width/2;
            int offset_y = params->padding_height/2;
            double*** out = (double***)previous_gradient->v;
            double*** in = (double***)gradient->v;
            for(int i=0;i<output_channels;i++)
            {
                double** matrix_out = out[i];
                double** matrix_in = in[i];
                for(int j=0;j<output_height;j++)
                {
                    int in_y=j+offset_y;
                    for(int k=0;k<output_width;k++)
                    {
                        int in_x=i+offset_x;
                        matrix_out[j][k]=matrix_in[in_y][in_x];
                    }
                }
            }
        }
    }
    return layer->previous_gradients;
}

//Flatten layer doesn't calculate anything and the function is not called  in backward loop
void backward_calculation_Padding2D(optimizer *optimizer, layer *layer, int layer_index)
{
}

void compile_layer_Padding2D(shape* input_shape, layer *layer)
{
    //Should be 3D shape with channels, height, width
    layer->input_shape = clone_shape(input_shape);
    padding2D_parameters* params = (padding2D_parameters*)layer->parameters;
    layer->parameters=params;
    layer->output_shape->sizes[0]= layer->input_shape->sizes[0];
    layer->output_shape->sizes[1]= layer->input_shape->sizes[1]+params->padding_height;
    layer->output_shape->sizes[2]= layer->input_shape->sizes[2]+params->padding_width;
}

void configure_layer_Padding2D(layer* layer)
{
    //Set used methods for the layer
    configure_default_layer(layer);
    layer->parameters = malloc(sizeof(padding2D_parameters));
    layer->init_training_memory = init_memory_training_no_activation;
    layer->clear_training_memory= clear_layer_training_memory_no_activation;
    layer->forward_propagation_training_loop= forward_propagation_training_loop_no_activation;
    layer->forward_calculation_predict=forward_calculation_predict_Padding2D;
    layer->forward_calculation_training=forward_calculation_training_Padding2D;
    layer->backward_propagation_loop=backward_propagation_loop_Padding2D;
    layer->backward_calculation=backward_calculation_Padding2D;
    layer->compile_layer=compile_layer_Padding2D;
    layer->read_parameters = read_parameters_Padding2D;
    layer->save_parameters = save_parameters_Padding2D;
}

layer* build_layer_Padding2D(int padding_height, int padding_width)
{
    layer *layer = (struct layer *)malloc(sizeof(struct layer));
    configure_layer_Padding2D(layer);
    padding2D_parameters* params = (padding2D_parameters*)layer->parameters;
    params->padding_height=padding_height;
    params->padding_width=padding_width;
    layer->type = PADDING2D;
    layer->activation = NULL;
    layer->output_shape = build_shape(ThreeD);
    return layer;
}