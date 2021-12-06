#include "../include/layer.h"

typedef struct maxpool2D_parameters{
    int pool_size;
    int stride;
} maxpool2D_parameters;

void save_parameters_MaxPooling2D(FILE* fp, layer* layer)
{
    maxpool2D_parameters* params = (maxpool2D_parameters*)layer->parameters;
    fprintf(fp, "pool_size:%d, stride:%d\n", params->pool_size, params->stride);
}

void read_parameters_MaxPooling2D(FILE* fp, layer* layer)
{
    maxpool2D_parameters* params = (maxpool2D_parameters*)layer->parameters;
    fscanf(fp, "pool_size:%d, stride:%d\n", &params->pool_size, &params->stride);
}

//Convolution calculation layer for training
tensor *forward_calculation_training_MaxPooling2D(const tensor *input, tensor *output, tensor* activation_input, layer *layer)
{
    maxpool2D_parameters* params = (maxpool2D_parameters*)layer->parameters;
    int output_width = layer->output_shape->sizes[2];
    int output_height = layer->output_shape->sizes[1];
    int output_channels = layer->output_shape->sizes[0];
    int input_channels = layer->output_shape->sizes[0];
    int pool_size = params->pool_size;
    int stride = params->stride;
    double*** cube_out = (double***)output->v;
    double*** cube_in = (double***)input->v;
    //Iterate trough each output height
    for(int i=0;i<output_height;i++)
    {
        int start_y=i*stride;
        int end_y=start_y+pool_size;
        //And trough each output width
        for(int j=0;j<output_width;j++)
        {
            //Calculates the associated matrix slice locations in input space
            int start_x=j*stride;
            int end_x=start_x+pool_size;
            //Iterate trough each output channel
            for(int c_out =0;c_out<output_channels;c_out++)
            {
                double** matrix_out = cube_out[c_out];
                double** matrix_in = cube_in[c_out];
                double max = - __DBL_MAX__;
                //Iterate trough each cell of input slice
                for(int i_y=start_y;i_y<end_y;i_y++)
                {
                    int i_pool_y = i_y-start_y;
                    for(int i_x=start_x;i_x<end_y;i_x++)
                    {
                        int i_pool_x = i_x-start_x;
                        double v = matrix_in[i_y][i_x];
                        max = v>max?v:max;
                    }
                }
                //Set max of the pool to output
                matrix_out[i][j]=max;
            }
        }
    }
    return output;
}

//Forward calculation function for Flatten layer when predicting
tensor *forward_calculation_predict_MaxPooling2D(const tensor *input, tensor *output, layer *layer)
{
    forward_calculation_training_MaxPooling2D(input, output, NULL, layer);
}

void compile_layer_MaxPooling2D(shape* input_shape, layer *layer)
{
    maxpool2D_parameters* params = (maxpool2D_parameters*)layer->parameters;
    int pool_size= params->pool_size;
    int stride = params->stride;
    //Storing input_shape (should be ThreeD) with img_channels, img_height, img_width
    layer->input_shape = clone_shape(input_shape);
    int output_height = (layer->input_shape->sizes[1]-pool_size)/stride + 1;
    int output_width = (layer->input_shape->sizes[2]-pool_size)/stride + 1;
    layer->output_shape->sizes[0]=layer->input_shape->sizes[0];
    layer->output_shape->sizes[1]=output_height;
    layer->output_shape->sizes[2]=output_width;
}

void configure_layer_MaxPooling2D(layer* layer)
{
    configure_default_layer(layer);
    maxpool2D_parameters* params = (maxpool2D_parameters*)malloc(sizeof(maxpool2D_parameters));
    layer->save_parameters = save_parameters_MaxPooling2D;
    layer->read_parameters = read_parameters_MaxPooling2D;
    layer->init_training_memory = init_memory_training_no_activation;
    layer->clear_training_memory= clear_layer_training_memory_no_activation;
    layer->forward_propagation_training_loop = forward_propagation_training_loop_no_activation;
    layer->compile_layer = compile_layer_MaxPooling2D;
    layer->forward_calculation_training = forward_calculation_training_MaxPooling2D;
    layer->forward_calculation_predict = forward_calculation_predict_MaxPooling2D;
    //TODO
}

layer* build_layer_MaxPooling2D(int pool_size, int stride)
{
    layer *layer = (struct layer *)malloc(sizeof(struct layer));
    configure_layer_MaxPooling2D(layer);
    maxpool2D_parameters* params = (maxpool2D_parameters*)layer->parameters;
    params->pool_size = pool_size;
    params->stride = stride;
    layer->type = MAXPOOL2D;
    layer->output_shape = build_shape(ThreeD);
    layer->activation = NULL;
    return layer;
}