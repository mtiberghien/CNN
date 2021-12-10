#include "../../include/layer.h"

typedef struct maxpool2D_parameters{
    int pool_height;
    int pool_width;
    int stride;
} maxpool2D_parameters;

void save_parameters_MaxPooling2D(FILE* fp, layer* layer)
{
    maxpool2D_parameters* params = (maxpool2D_parameters*)layer->parameters;
    fprintf(fp, "pool_width:%d, pool_height:%d, stride:%d\n", params->pool_width, params->pool_height, params->stride);
}

void read_parameters_MaxPooling2D(FILE* fp, layer* layer)
{
    maxpool2D_parameters* params = (maxpool2D_parameters*)layer->parameters;
    fscanf(fp, "pool_width:%d, pool_height:%d, stride:%d\n", &params->pool_width, &params->pool_height, &params->stride);
}

//Convolution calculation layer for training
tensor *forward_calculation_training_MaxPooling2D(const tensor *input, tensor *output, tensor* activation_input, layer *layer)
{
    maxpool2D_parameters* params = (maxpool2D_parameters*)layer->parameters;
    int output_width = layer->output_shape->sizes[2];
    int output_height = layer->output_shape->sizes[1];
    int output_channels = layer->output_shape->sizes[0];
    int pool_height = params->pool_height;
    int pool_width = params->pool_width;
    int stride = params->stride;
    double*** cube_out = (double***)output->v;
    double*** cube_in = (double***)input->v;
    //Iterate trough each output channel
    for(int c_out =0;c_out<output_channels;c_out++)
    {
        double** matrix_out = cube_out[c_out];
        double** matrix_in = cube_in[c_out];
        for(int i=0;i<output_height;i++)
        {
            int start_y=i*stride;
            int end_y=start_y+pool_height;
            double* array_out = matrix_out[i];
            //And trough each output width
            for(int j=0;j<output_width;j++)
            {
                //Calculates the associated matrix slice locations in input space
                int start_x=j*stride;
                int end_x=start_x+pool_width;
                double max = - __DBL_MAX__;
                //Iterate trough each cell of input slice
                for(int i_y=start_y;i_y<end_y;i_y++)
                {
                    double* array_in = matrix_in[i_y];
                    for(int i_x=start_x;i_x<end_x;i_x++)
                    {
                        double v = array_in[i_x];
                        max = v>max?v:max;
                    }
                }
                //Set max of the pool to output
                array_out[j]=max;
            }
        }
    }
    //Iterate trough each output height
    return output;
}

//Forward calculation function for Flatten layer when predicting
tensor *forward_calculation_predict_MaxPooling2D(const tensor *input, tensor *output, layer *layer)
{
    forward_calculation_training_MaxPooling2D(input, output, NULL, layer);
}

tensor *backward_propagation_loop_MaxPooling2D(tensor *gradients, optimizer *optimizer, struct layer *layer, int layer_index)
{
    maxpool2D_parameters* params = (maxpool2D_parameters*)layer->parameters;
    int output_channels = layer->output_shape->sizes[0];
    int output_height = layer->output_shape->sizes[1];
    int output_width = layer->output_shape->sizes[2];
    int batch_size = layer->batch_size;
    int pool_height = params->pool_height;
    int pool_width = params->pool_width;
    int stride = params->stride;
    #pragma omp parallel for
    for(int i=0;i<batch_size;i++)
    {
        double*** cube_in = (double***)layer->layer_inputs[i].v;
        double*** cube_gradient_previous = (double***)layer->previous_gradients[i].v;
        double*** cube_gradient=(double***)gradients[i].v;
         //Iterate trough each output channel
        for(int c_out =0;c_out<output_channels;c_out++)
        {
            double** matrix_in = cube_in[c_out];
            double** matrix_gradient_previous = cube_gradient_previous[c_out];
            double** matrix_gradient = cube_gradient[c_out];
            for(int out_y=0;out_y<output_height;out_y++)
            {
                int start_y=out_y*stride;
                int end_y=start_y+pool_height;
                double* array_gradient=matrix_gradient[out_y];
                for(int out_x=0;out_x<output_width;out_x++)
                {
                    int start_x=out_x*stride;
                    int end_x=start_x+pool_width;
                    double max = - __DBL_MAX__;
                    int x_max=0;
                    int y_max=0;
                    //Iterate trough each cell of input slice
                    for(int i_y=start_y;i_y<end_y;i_y++)
                    {
                        double* array_in= matrix_in[i_y];
                        for(int i_x=start_x;i_x<end_x;i_x++)
                        {
                            double v = array_in[i_x];
                            if(v>max)
                            {
                                max = v;
                                x_max=i_x;
                                y_max=i_y;
                            }
                        }
                    }
                    if(layer_index>=0)
                    {
                        matrix_gradient_previous[y_max][x_max]=array_gradient[out_x];
                    }
                    array_gradient[out_x]=0;
                }
            } 
        }       
    }
    return layer->previous_gradients;
}

//MaxPooling2D layer doesn't calculate anything and the function is not called  in backward loop
void backward_calculation_MaxPooling2D(optimizer *optimizer, layer *layer, int layer_index)
{
}

void compile_layer_MaxPooling2D(shape* input_shape, layer *layer)
{
    maxpool2D_parameters* params = (maxpool2D_parameters*)layer->parameters;
    int pool_height = params->pool_height;
    int pool_width = params->pool_width;
    int stride = params->stride;
    //Storing input_shape (should be ThreeD) with img_channels, img_height, img_width
    layer->input_shape = clone_shape(input_shape);
    int output_height = (layer->input_shape->sizes[1]-pool_height)/stride + 1;
    int output_width = (layer->input_shape->sizes[2]-pool_width)/stride + 1;
    layer->output_shape->sizes[0]=layer->input_shape->sizes[0];
    layer->output_shape->sizes[1]=output_height;
    layer->output_shape->sizes[2]=output_width;
}

void configure_layer_MaxPooling2D(layer* layer)
{
    configure_default_layer(layer);
    maxpool2D_parameters* params = (maxpool2D_parameters*)malloc(sizeof(maxpool2D_parameters));
    layer->parameters=params;
    layer->save_parameters = save_parameters_MaxPooling2D;
    layer->read_parameters = read_parameters_MaxPooling2D;
    layer->init_training_memory = init_memory_training_no_activation;
    layer->clear_training_memory= clear_layer_training_memory_no_activation;
    layer->forward_propagation_training_loop = forward_propagation_training_loop_no_activation;
    layer->compile_layer = compile_layer_MaxPooling2D;
    layer->forward_calculation_training = forward_calculation_training_MaxPooling2D;
    layer->forward_calculation_predict = forward_calculation_predict_MaxPooling2D;
    layer->backward_propagation_loop = backward_propagation_loop_MaxPooling2D;
    layer->backward_calculation = backward_calculation_MaxPooling2D;
}

layer* build_layer_MaxPooling2D(int pool_height, int pool_width, int stride)
{
    layer *layer = (struct layer *)malloc(sizeof(struct layer));
    configure_layer_MaxPooling2D(layer);
    maxpool2D_parameters* params = (maxpool2D_parameters*)layer->parameters;
    params->pool_height = pool_height;
    params->pool_width = pool_width;
    params->stride = stride;
    layer->type = MAXPOOL2D;
    layer->output_shape = build_shape(ThreeD);
    layer->activation = NULL;
    return layer;
}