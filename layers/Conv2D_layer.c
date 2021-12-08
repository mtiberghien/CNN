#include "../include/layer.h"
#include "../include/tensor.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

typedef struct conv2D_parameters{
    short padding;
    int stride;
    int kernel_width;
    int kernel_height;
    int n_output_channels;
    tensor* filters;
    tensor* filters_gradients;
    tensor biases;
    tensor biases_gradients;
    struct layer* padding_layer;
} conv2D_parameters;

void build_shape_list_Conv2D(layer* layer, shape_list* shape_list)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    shape_list->n_shapes=2;
    shape_list->shapes=malloc(sizeof(shape)*2);
    shape_list->shapes[0]=*clone_shape(params->biases.shape);
    shape_list->shapes[1]=*clone_shape(params->filters->shape);
}

void clear_parameters_Conv2D(layer* layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    clear_tensors(params->filters, params->n_output_channels);
    free(params->filters);
    clear_tensor(&params->biases);
    if(params->padding)
    {
        clear_layer(params->padding_layer);
        free(params->padding_layer);
    }
}

int get_trainable_parameters_count_Conv2D(layer* layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    int total = params->n_output_channels;
    for(int i=0;i<params->n_output_channels;i++)
    {
        total += params->kernel_width*params->kernel_height*layer->input_shape->sizes[0];
    }
    return total;
}

void save_parameters_Conv2D(FILE* fp, layer* layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    fprintf(fp, "n_output_channels:%d, kernel_width:%d, kernel_height:%d, stride:%d, padding:%hd\n", params->n_output_channels, params->kernel_width, params->kernel_height, params->stride, params->padding);
}

void save_trainable_parameters_Conv2D(FILE* fp, layer* layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    for(int i=0;i<params->n_output_channels;i++)
    {
        save_tensor(fp, &params->filters[i]);
    }
    save_tensor(fp, &params->biases);
}

void read_parameters_Conv2D(FILE* fp, layer* layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    fscanf(fp, "n_output_channels:%d, kernel_width:%d, kernel_height:%d, stride:%d, padding:%hd\n", &params->n_output_channels, &params->kernel_width, &params->kernel_height, &params->stride, &params->padding);
}

void read_trainable_parameters_Conv2D(FILE* fp, layer* layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    for(int i=0;i<params->n_output_channels;i++)
    {
        read_tensor(fp, &params->filters[i]);
    }
    read_tensor(fp, &params->biases);
}

//Default forward propagation loop
tensor *forward_propagation_training_loop_Conv2D(const tensor *inputs, int batch_size, struct layer *layer, progression* progression)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    if(params->padding)
    {
        inputs =params->padding_layer->forward_propagation_training_loop(inputs, batch_size, params->padding_layer, NULL);
    }
    forward_propagation_training_loop(inputs, batch_size, layer, progression);
}

tensor *forward_propagation_predict_loop_Conv2D(const tensor *inputs, int batch_size, struct layer *layer, progression* progression)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    if(params->padding)
    {
        inputs =params->padding_layer->forward_propagation_predict_loop(inputs, batch_size, params->padding_layer, NULL);
    }
    forward_propagation_predict_loop(inputs, batch_size,layer, progression);
}

//Convolution calculation layer for training
tensor *forward_calculation_training_Conv2D(const tensor *input, tensor *output, tensor* activation_input, layer *layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    int output_width = layer->output_shape->sizes[2];
    int output_height = layer->output_shape->sizes[1];
    int output_channels = layer->output_shape->sizes[0];
    int input_channels = layer->input_shape->sizes[0];
    int kernel_width = params->kernel_width;
    int kernel_height = params->kernel_height;
    int stride = params->stride;
    double*** cube_out = (double***)output->v;
    double*** cube_in = (double***)input->v;
    double*** cube_activation_in = (double***)activation_input->v;
    //Iterate trough each output channel
    for(int c_out =0;c_out<output_channels;c_out++)
    {
        double** matrix_out = cube_out[c_out];
        double** matrix_activation_in = cube_activation_in[c_out];
        double*** filters = (double***)params->filters[c_out].v;
        //Iterate trough each output height
        for(int i=0;i<output_height;i++)
        {
            //Calculates the associated matrix slice height in input space
            int start_y=i*stride;
            int end_y=start_y+kernel_height;
            double* array_out = matrix_out[i];
            double* array_activation_in = matrix_activation_in[i];
            //Iterate trough each output width
            for(int j=0;j<output_width;j++)
            {
                //Calculates the associated matrix slice width in input space
                int start_x=j*stride;
                int end_x=start_x+kernel_width;
                //Iterate trough each input channel
                for(int c_in=0;c_in<input_channels;c_in++)
                {
                    double** matrix_filter = filters[c_in];
                    double** matrix_in = cube_in[c_in];
                    //Iterate trough each cell of input slice
                    for(int i_y=start_y;i_y<end_y;i_y++)
                    {
                        int i_kernel_y = i_y-start_y;
                        double* array_filter=matrix_filter[i_kernel_y];
                        double* array_in=matrix_in[i_y];
                        for(int i_x=start_x;i_x<end_x;i_x++)
                        {
                            int i_kernel_x = i_x-start_x;
                            //Sum the product of each input channel with associated output channel filter
                            array_out[j]+=array_filter[i_kernel_x]*array_in[i_x];
                        }
                    }
                }
                //Add biases to each channel
                array_out[j]+=params->biases.v[c_out];
                //Copyt value of output before activation (used in back propagation)
                array_activation_in[j]=array_out[j];
            }
        }
    }
    if (layer->activation)
    {
        //Execute activation function and return output tensor
        output = layer->activation->activation_forward(output, layer->activation);
    }
    return output;
}

//Convolution calculation layer for prediction only
tensor *forward_calculation_predict_Conv2D(const tensor *input, tensor *output, layer *layer)
{
   conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    int output_width = layer->output_shape->sizes[2];
    int output_height = layer->output_shape->sizes[1];
    int output_channels = layer->output_shape->sizes[0];
    int input_channels = layer->input_shape->sizes[0];
    int kernel_width = params->kernel_width;
    int kernel_height = params->kernel_height;
    int stride = params->stride;
    double*** cube_out = (double***)output->v;
    double*** cube_in = (double***)input->v;
    //Iterate trough each output channel
    for(int c_out =0;c_out<output_channels;c_out++)
    {
        double** matrix_out = cube_out[c_out];
        double*** filters = (double***)params->filters[c_out].v;
        //Iterate trough each output height
        for(int i=0;i<output_height;i++)
        {
            //Calculates the associated matrix slice height in input space
            int start_y=i*stride;
            int end_y=start_y+kernel_height;
            double* array_out = matrix_out[i];
            //Iterate trough each output width
            for(int j=0;j<output_width;j++)
            {
                //Calculates the associated matrix slice width in input space
                int start_x=j*stride;
                int end_x=start_x+kernel_width;
                //Iterate trough each input channel
                for(int c_in=0;c_in<input_channels;c_in++)
                {
                    double** matrix_filter = filters[c_in];
                    double** matrix_in = cube_in[c_in];
                    //Iterate trough each cell of input slice
                    for(int i_y=start_y;i_y<end_y;i_y++)
                    {
                        int i_kernel_y = i_y-start_y;
                        double* array_filter=matrix_filter[i_kernel_y];
                        double* array_in=matrix_in[i_y];
                        for(int i_x=start_x;i_x<end_x;i_x++)
                        {
                            int i_kernel_x = i_x-start_x;
                            //Sum the product of each input channel with associated output channel filter
                            array_out[j]+=array_filter[i_kernel_x]*array_in[i_x];
                        }
                    }
                }
                //Add biases to each channel
                array_out[j]+=params->biases.v[c_out];
            }
        }
    }
    if (layer->activation)
    {
        //Execute activation function and return output tensor
        output = layer->activation->activation_forward(output, layer->activation);
    }
    return output;
}

tensor *backward_propagation_loop_Conv2D(tensor *gradients, optimizer *optimizer, struct layer *layer, int layer_index)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    int output_width = layer->output_shape->sizes[2];
    int output_height = layer->output_shape->sizes[1];
    int output_channels = layer->output_shape->sizes[0];
    int input_channels = layer->input_shape->sizes[0];
    int batch_size = layer->batch_size;
    int kernel_width = params->kernel_width;
    int kernel_height = params->kernel_height;
    int stride = params->stride;
    double* biases_gradients= (double*)params->biases_gradients.v;
    //#pragma omp parallel for
    for(int i=0;i<batch_size;i++)
    {
        tensor* gradient = &gradients[i];       
        tensor* output = &layer->outputs[i];
        double*** cube_gradient_previous = (double***)layer->previous_gradients[i].v;
        const tensor* input = &layer->layer_inputs[i];
        double*** cube_input = (double***)input->v;
        if (layer->activation)
        {
            //Back propagate the gradient error tensor
            gradient = layer->activation->activation_backward_propagation(&layer->activation_input[i], gradient, output, layer->activation);
        }
        for(int c_out=0;c_out<output_channels;c_out++)
        {
            double*** cube_filter = (double***)params->filters[c_out].v;
            double*** cube_filter_gradient = (double***) params->filters_gradients[c_out].v;
            double** matrix_gradient = ((double***)gradient->v)[c_out];
            //Iterate trough each output height
            for(int i=0;i<kernel_height;i++)
            {
                int start_y=i*stride;
                int end_y=start_y+output_height;
                //And trough each output width
                for(int j=0;j<kernel_width;j++)
                {
                    //Calculates the associated matrix slice locations in input space
                    int start_x=j*stride;
                    int end_x=start_x+output_width;
                    //Iterate trough each input channel
                    for(int c_in=0;c_in<input_channels;c_in++)
                    {
                        double* array_filter_gradient = cube_filter_gradient[c_in][i];
                        double** matrix_in = cube_input[c_in];
                        double** matrix_gradient_previous= cube_gradient_previous[c_in];
                        double filter_i_j = cube_filter[c_in][i][j];
                        //Iterate trough each cell of input slice
                        for(int i_y=start_y;i_y<end_y;i_y++)
                        {
                            int i_gradient_y = i_y-start_y;
                            double* array_in = matrix_in[i_y];
                            double* array_gradient_previous= matrix_gradient_previous[i_y];
                            double* array_gradient = matrix_gradient[i_gradient_y];
                            for(int i_x=start_x;i_x<end_x;i_x++)
                            {
                                int i_gradient_x = i_x-start_x;
                                double gradient_y_x = array_gradient[i_gradient_x];
                                //Sum the product of each input channel with associated output gradient into the filter_gradient
                                array_filter_gradient[j]+=gradient_y_x*array_in[i_x];
                                if(layer_index>0)
                                {
                                    array_gradient_previous[i_x]+=gradient_y_x*filter_i_j;
                                }                               
                            }
                        }
                    }
                }
            }
            //Iterate trough each output height
            for(int i=0;i<output_height;i++)
            {
                int start_y=i*stride;
                int end_y=start_y+kernel_height;
                //And trough each output width
                for(int j=0;j<output_width;j++)
                {
                    double gradient_ij = matrix_gradient[i][j];
                    //biases_gradient is the sum of batch gradients
                    biases_gradients[c_out]+=gradient_ij;
                    //Reset gradient for next episode
                    matrix_gradient[i][j]=0;
                }
            }
        }
    }
    layer->backward_calculation(optimizer, layer, layer_index);
    return params->padding? params->padding_layer->backward_propagation_loop(layer->previous_gradients, optimizer, params->padding_layer, 0): layer->previous_gradients;
}

//Backward propagation function for Fully Connected layer (perceptron)
void backward_calculation_Conv2D(optimizer *optimizer, layer *layer, int layer_index)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    int output_channel = layer->output_shape->sizes[0];
    int input_channel = layer->input_shape->sizes[0];
    int output_height = layer->output_shape->sizes[1];
    int kernel_width = params->kernel_width;
    int kernel_height = params->kernel_height;
    double* biases_gradient = params->biases_gradients.v;
    //Update biases using new gradient
    for (int c_out = 0; c_out < output_channel; c_out++)
    {
        tensor* filter = &params->filters[c_out];
        int* iterator = get_iterator(filter);
        double*** cube_filters = (double***)params->filters[c_out].v;
        double*** cube_filters_gradient = (double***)params->filters_gradients[c_out].v;
        //Optimizer update bias
        params->biases.v[c_out] = optimizer->apply_gradient(params->biases.v[c_out], biases_gradient[c_out], layer_index, 0, &c_out, optimizer);
        //Reset gradient for next episode
        biases_gradient[c_out]=0;
        for(int c_in=0;c_in<input_channel;c_in++)
        {
            double** matrix_filters = cube_filters[c_in];
            double** matrix_filters_gradient = cube_filters_gradient[c_in];
            //Calculate the gradient for previous layer and update weights
            for(int i_y=0;i_y<kernel_height;i_y++)
            {
                double* array_filters = matrix_filters[i_y];
                double* array_filters_gradient = matrix_filters_gradient[i_y];
                for(int i_x=0;i_x<kernel_width;i_x++)
                {
                    //Sum the product of each input channel with associated output channel filter
                    optimizer->apply_gradient(array_filters[i_x], array_filters_gradient[i_x], layer_index, 1, iterator, optimizer);
                    iterator = filter->get_next(filter, iterator);
                    //Reset filters gradient for next episode
                    array_filters_gradient[i_x]=0;
                }
            }
        }
        free(iterator);
    }
}

void init_memory_predict_Conv2D(layer* layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    init_memory_predict(layer);
    if(params->padding)
    {
        params->padding_layer->init_predict_memory(params->padding_layer);
    }
}

void clear_memory_predict_Conv2D(layer* layer)
{
   conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
   clear_layer_predict_memory(layer);
   if(params->padding)
   {
       params->padding_layer->clear_predict_memory(params->padding_layer);
   }
}

void init_memory_training_Conv2D(layer* layer)
{
    init_memory_training(layer);
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    params->filters_gradients=malloc(sizeof(tensor)*params->n_output_channels);
    for(int i=0;i<params->n_output_channels;i++)
    {
        initialize_tensor(&params->filters_gradients[i], params->filters->shape);
    }
    initialize_tensor(&params->biases_gradients, params->biases.shape);
    if(params->padding)
    {
        params->padding_layer->init_training_memory(params->padding_layer);
    }
}

void clear_layer_training_memory_Conv2D(layer *layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    clear_layer_training_memory(layer);
    clear_tensors(params->filters_gradients, params->n_output_channels);
    free(params->filters_gradients);
    clear_tensor(&params->biases_gradients);
    if(params->padding)
    {
        params->padding_layer->clear_training_memory(params->padding_layer);
    }
}


void compile_layer_Conv2D(shape* input_shape, layer *layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    if(params->padding)
    {
        params->padding_layer->compile_layer(input_shape, params->padding_layer);
        input_shape = params->padding_layer->output_shape;
    }
    int kernel_width = params->kernel_width;
    int kernel_height = params->kernel_height;
    short padding = params->padding;
    int stride = params->stride;
    //Storing input_shape (should be ThreeD) with img_channels, img_height, img_width
    layer->input_shape = clone_shape(input_shape);
    //Calculating output height and width according to input_shape and parameters using ((Is - K + (K-1)*P)/St)+1
    int output_height = ((input_shape->sizes[1] - kernel_height + (kernel_height-1)*padding)/stride)+1;
    int output_width = ((input_shape->sizes[2] - kernel_width +(kernel_width-1)*padding)/stride)+1;
    layer->output_shape->sizes[1]= output_height;
    layer->output_shape->sizes[2]= output_width;
    //Filter shape is created according to kernel_size and input_channel size
    shape* filter_shape = build_shape(ThreeD);
    filter_shape->sizes[0]=input_shape->sizes[0];
    filter_shape->sizes[1]=params->kernel_height;
    filter_shape->sizes[2]=params->kernel_width;
    //Filters initialization
    double invert_rand_max = (double)1.0 / (double)RAND_MAX;
    int fan_in = input_shape->sizes[0]*input_shape->sizes[1]*input_shape->sizes[2];
    // glorot uniform init: https://github.com/ElefHead/numpy-cnn/blob/master/utilities/initializers.py
    double limit = sqrt((double)6 /(fan_in + (params->kernel_width*params->kernel_height*params->n_output_channels)));
    params->filters=malloc(sizeof(tensor)*params->n_output_channels);
    for(int i=0;i<params->n_output_channels;i++)
    {
        tensor* filter = &params->filters[i];
        initialize_tensor(filter, filter_shape);
        int* iterator = get_iterator(filter);
        while(!filter->is_done(filter, iterator))
        {
            double v = (2 * limit * ((double)rand() * invert_rand_max)) - limit;
            filter->set_value(filter, iterator, v);
            iterator = filter->get_next(filter, iterator);
        }
        free(iterator);
    }
    clear_shape(filter_shape);
    free(filter_shape);
    shape* biases_shape = build_shape(OneD);
    biases_shape->sizes[0]=params->n_output_channels;
    initialize_tensor(&params->biases, biases_shape);
    clear_shape(biases_shape);
    free(biases_shape);
    for (int i = 0; i < params->n_output_channels; i++)
    {
        params->biases.v[i] = (2 * limit * ((double)rand() * invert_rand_max)) - limit;
    }
}

void configure_layer_Conv2D(layer* layer)
{
    configure_default_layer(layer);
    conv2D_parameters* params = (conv2D_parameters*)malloc(sizeof(conv2D_parameters));
    layer->parameters = params;
    layer->compile_layer = compile_layer_Conv2D;
    layer->init_training_memory = init_memory_training_Conv2D;
    layer->clear_training_memory = clear_layer_training_memory_Conv2D;
    layer->init_predict_memory = init_memory_predict_Conv2D;
    layer->clear_predict_memory = clear_memory_predict_Conv2D;
    layer->forward_calculation_training = forward_calculation_training_Conv2D;
    layer->forward_calculation_predict = forward_calculation_predict_Conv2D;
    layer->backward_calculation = backward_calculation_Conv2D;
    layer->build_shape_list = build_shape_list_Conv2D;
    layer->clear_parameters = clear_parameters_Conv2D;
    layer->read_parameters = read_parameters_Conv2D;
    layer->save_parameters = save_parameters_Conv2D;
    layer->read_trainable_parameters = read_trainable_parameters_Conv2D;
    layer->save_trainable_parameters = save_trainable_parameters_Conv2D;
    layer->forward_propagation_predict_loop = forward_propagation_predict_loop_Conv2D;
    layer->forward_propagation_training_loop= forward_propagation_training_loop_Conv2D;
    layer->backward_propagation_loop = backward_propagation_loop_Conv2D;
    layer->get_trainable_parameters_count = get_trainable_parameters_count_Conv2D;
}

layer* build_layer_Conv2D(int output_channel_size, int kernel_width, int kernel_height, int stride, short padding, activation* activation)
{
    layer *layer = (struct layer *)malloc(sizeof(struct layer));
    configure_layer_Conv2D(layer);
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    params->n_output_channels = output_channel_size;
    params->kernel_width = kernel_width;
    params->kernel_height = kernel_height;
    params->stride = stride;
    params->padding = padding > 0;
    if(params->padding)
    {
        params->padding_layer = build_layer_Padding2D(params->kernel_height-1, params->kernel_width-1);
    }
    layer->type = CONV2D;
    layer->output_shape = build_shape(ThreeD);
    layer->output_shape->sizes[0]=output_channel_size;
    layer->activation = activation;
    return layer;
}