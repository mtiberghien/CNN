#include "include/layer.h"
#include "include/tensor.h"
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

typedef struct conv2D_parameters{
    short padding;
    int stride;
    int kernel_width;
    int kernel_height;
    int n_output_channels;
    tensor filters;
    tensor filters_gradients;
    tensor biases;
    tensor biases_gradients;
} conv2D_parameters;

//Clear memory of temporary stored inputs and outputs
void clear_layer_training_memory(layer *layer)
{
    #pragma omp parallel for
    for (int i = 0; i < layer->batch_size; i++)
    {
        clear_tensor(&layer->outputs[i]);
        clear_tensor(&layer->activation_input[i]);
        clear_tensor(&layer->previous_gradients[i]);
    }
    free(layer->previous_gradients);
    free(layer->activation_input);
    free(layer->layer_inputs);
    free(layer->outputs);
}

//Clear memory of temporary stored inputs and outputs
void clear_layer_training_memory_FC(layer *layer)
{
    fc_parameters* params = (fc_parameters*)layer->parameters;
    clear_layer_training_memory(layer);
    clear_tensor(&params->weights_gradients);
    clear_tensor(&params->biases_gradients);
}

void clear_layer_predict_memory(layer* layer)
{
    #pragma omp parallel for
    for (int i = 0; i < layer->batch_size; i++)
    {
        clear_tensor(&layer->outputs[i]);
    }
    free(layer->outputs);
}

void clear_parameters_FC(layer* layer)
{
    fc_parameters* params = (fc_parameters*)layer->parameters;
    clear_tensor(&params->weights);
    clear_tensor(&params->biases);
}

void clear_layer(layer *layer)
{
    layer->clear_parameters(layer);
    if (layer->activation)
    {
        free(layer->activation);
    }
    clear_shape(layer->input_shape);
    clear_shape(layer->output_shape);
    free(layer->output_shape);
    free(layer->input_shape);
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

void init_memory_training(layer* layer)
{
    shape* input_shape = layer->input_shape;
    shape* output_shape = layer->output_shape;
    int batch_size = layer->batch_size;
    layer->layer_inputs=(tensor*) malloc(sizeof(tensor)*batch_size);
    layer->activation_input = (tensor *)malloc(sizeof(tensor)*batch_size);
    layer->previous_gradients = (tensor *)malloc(sizeof(tensor)*batch_size);
    layer->outputs = malloc(sizeof(tensor) * batch_size);
    #pragma omp parallel for
    for(int i=0;i<batch_size;i++)
    {
        initialize_tensor(&layer->outputs[i], output_shape);
        initialize_tensor(&layer->activation_input[i], output_shape);
        initialize_tensor(&layer->previous_gradients[i], input_shape);
    }
}

void init_memory_training_FC(layer* layer)
{
    init_memory_training(layer);
    fc_parameters* params = (fc_parameters*)layer->parameters;
    initialize_tensor(&params->weights_gradients, params->weights.shape);
    initialize_tensor(&params->biases_gradients, layer->output_shape);
}

void init_memory_predict(layer* layer)
{
    int batch_size = layer->batch_size;
    shape* output_shape = layer->output_shape;
    layer->outputs = malloc(sizeof(tensor) * batch_size);
    #pragma omp parallel for
    for(int i=0;i<batch_size;i++)
    {
        initialize_tensor(&layer->outputs[i], output_shape);
    }
}

void build_shape_list_FC(layer* layer, shape_list* shape_list)
{
    fc_parameters* params = (fc_parameters*)layer->parameters;
    shape_list->n_shapes=2;
    shape_list->shapes=malloc(sizeof(shape)*2);
    shape_list->shapes[0]=*clone_shape(params->biases.shape);
    shape_list->shapes[1]=*clone_shape(layer->output_shape);
}

//Default forward propagation loop
tensor *forward_propagation_training_loop(const tensor *inputs, int batch_size, struct layer *layer, progression* progression)
{
    // Loop into input batch
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++)
    {
        //Output tensor memory allocation
        tensor *output = &layer->outputs[i];
        tensor* activation_input = &layer->activation_input[i];
        const tensor* input = &inputs[i];
        layer->layer_inputs[i]=inputs[i];
        //Execute specific forward propagation
        layer->forward_calculation_training(input, output, activation_input, layer);
    }
    return layer->outputs;
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

tensor *forward_propagation_predict_loop(const tensor *inputs, int batch_size, struct layer *layer, progression* progression)
{
    // Loop into input batch
    #pragma omp parallel for shared(progression)
    for (int i = 0; i < batch_size; i++)
    {
        //Output tensor memory allocation
        tensor *output = &layer->outputs[i];
        //Execute specific forward propagation
        layer->forward_calculation_predict(&inputs[i], output, layer);
        if(progression)
        {
            progression->call_back(progression);
        }
    }
    return layer->outputs;
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

void save_parameters_FC(FILE* fp, layer* layer)
{
    fc_parameters* params = (fc_parameters*)layer->parameters;
    save_tensor(fp, &params->weights);
    save_tensor(fp, &params->biases);
}

void read_parameters_FC(FILE* fp, layer* layer)
{
    fc_parameters* params = (fc_parameters*)layer->parameters;
    read_tensor(fp, &params->weights);
    read_tensor(fp, &params->biases);
}

void save_layer(FILE *fp, layer *layer)
{
    fprintf(fp, "input_shape:");
    save_shape(fp, layer->input_shape);
    fprintf(fp, ", output_shape:");
    save_shape(fp, layer->output_shape);
    fprintf(fp, ", type:%d\n", layer->type);
    layer->save_parameters(fp, layer);
    save_activation(fp, layer->activation);
}

void build_shape_list_Conv2D(layer* layer, shape_list* shape_list)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    shape_list->n_shapes=2;
    shape_list->shapes=malloc(sizeof(shape)*2);
    shape_list->shapes[0]=*clone_shape(params->biases.shape);
    shape_list->shapes[1]=*clone_shape(layer->output_shape);
}

void clear_parameters_Conv2D(layer* layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    clear_tensor(&params->filters);
    clear_tensor(&params->biases);
}

void save_parameters_Conv2D(FILE* fp, layer* layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    fprintf(fp, "n_output_channels:%d, kernel_width:%d, kernel_height:%d stride:%d, padding:%hd\n", params->n_output_channels, params->kernel_width, params->kernel_height, params->stride, params->padding);
    save_tensor(fp, &params->filters);
    save_tensor(fp, &params->biases);
}

void read_parameters_Conv2D(FILE* fp, layer* layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    fscanf(fp, "n_output_channels:%d, kernel_width:%d, kernel_height:%d stride:%d, padding:%hd\n", &params->n_output_channels, &params->kernel_width, &params->kernel_height, &params->stride, &params->padding);
    read_tensor(fp, &params->filters);
    read_tensor(fp, &params->biases);
}

void unpad_gradient_previous(tensor* gradient_previous, tensor* p_gradient_previous, layer* layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    int padding_width = (params->kernel_width-1);
    int padding_height = (params->kernel_height-1);
    int output_height = gradient_previous->shape->sizes[1];
    int output_width = gradient_previous->shape->sizes[2];
    int output_channels = gradient_previous->shape->sizes[0];
    int offset_x = padding_width/2;
    int offset_y = padding_height/2;
    double*** out = (double***)gradient_previous->v;
    double*** in = (double***)p_gradient_previous->v;
    for(int i=0;i<output_channels;i++)
    {
        double** matrix_out = out[i];
        double** matrix_in = in[i];
        for(int j=0;j<output_height;j++)
        {
            for(int k=0;k<output_width;k++)
            {
                matrix_out[j][k]=matrix_in[j+offset_y][k+offset_x];
            }
        }
    }
}

tensor* pad_input(const tensor* input, layer* layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    tensor* result = malloc(sizeof(tensor));
    shape* shape = build_shape(ThreeD);
    int padding_width = (params->kernel_width-1);
    int padding_height = (params->kernel_height-1);
    shape->sizes[0]=input->shape->sizes[0];
    shape->sizes[1]=input->shape->sizes[1] + padding_height;
    shape->sizes[2]=input->shape->sizes[2] + padding_width;
    int input_height = input->shape->sizes[1];
    int input_width = input->shape->sizes[2];
    int input_channels = input->shape->sizes[0];
    int offset_x = padding_width/2;
    int offset_y = padding_height/2;
    initialize_tensor(result, shape);
    double*** out = (double***)result->v;
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
    clear_shape(shape);
    free(shape);
    return result;
}

//Convolution calculation layer for training
tensor *forward_calculation_training_Conv2D(const tensor *input, tensor *output, tensor* activation_input, layer *layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    int output_width = layer->output_shape->sizes[2];
    int output_height = layer->output_shape->sizes[1];
    int output_channels = layer->output_shape->sizes[0];
    int input_channels = layer->output_shape->sizes[0];
    int kernel_width = params->kernel_width;
    int kernel_height = params->kernel_height;
    int stride = params->stride;
    double*** filters = (double***)params->filters.v;
    tensor* p_input = params->padding? pad_input(input, layer):NULL;
    const tensor* used_input = params->padding? p_input:input;
    double*** cube_out = (double***)output->v;
    double*** cube_in = (double***)used_input->v;
    double*** cube_activation_in = (double***)activation_input->v;
    //Iterate trough each output height
    for(int i=0;i<output_height;i++)
    {
        //And trough each output width
        for(int j=0;j<output_width;j++)
        {
            //Calculates the associated matrix slice locations in input space
            int start_x=j*stride;
            int start_y=i*stride;
            int end_x=start_x+kernel_width;
            int end_y=start_y+kernel_height;
            //Iterate trough each output channel
            for(int c_out =0;c_out<output_channels;c_out++)
            {
                double** matrix_out = cube_out[c_out];
                double** matrix_activation_in = cube_activation_in[c_out];
                double** matrix_filter = filters[c_out];
                //Iterate trough each input channel
                for(int c_in=0;c_in<input_channels;c_in++)
                {
                    double** matrix_in = cube_in[c_in];
                    //Iterate trough each cell of input slice
                    for(int i_y=start_y;i_y<end_y;i_y++)
                    {
                        for(int i_x=start_x;i_x<end_y;i_x++)
                        {
                            //Sum the product of each input channel with associated output channel filter
                            matrix_out[i][j]+=matrix_filter[i_y-start_y][i_x-start_x]*matrix_in[i_y][i_x];
                        }
                    }
                }
                //Add biases to each channel
                matrix_out[i][j]+=params->biases.v[c_out];
                //Copyt value of output before activation (used in back propagation)
                matrix_activation_in[i][j]=matrix_out[i][j];
            }
        }
    }
    if(params->padding)
    {
        clear_tensor(p_input);
        free(p_input);
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
    int input_channels = layer->output_shape->sizes[0];
    int kernel_width = params->kernel_width;
    int kernel_height = params->kernel_height;
    int stride = params->stride;
    double*** filters = (double***)params->filters.v;
    tensor* p_input = params->padding? pad_input(input, layer):NULL;
    const tensor* used_input = params->padding? p_input:input;
    double*** cube_out = (double***)output->v;
    double*** cube_in = (double***)used_input->v;
    //Iterate trough each output height
    for(int i=0;i<output_height;i++)
    {
        //And trough each output width
        for(int j=0;j<output_width;j++)
        {
            //Calculates the associated matrix slice locations in input space
            int start_x=j*stride;
            int start_y=i*stride;
            int end_x=start_x+kernel_width;
            int end_y=start_x+kernel_height;
            //Iterate trough each output channel
            for(int c_out =0;c_out<output_channels;c_out++)
            {
                double** matrix_out = cube_out[c_out];
                double** matrix_filter = filters[c_out];
                //Iterate trough each input channel
                for(int c_in=0;c_in<input_channels;c_in++)
                {
                    double** matrix_in = cube_in[c_in];
                    //Iterate trough each cell of input slice
                    for(int i_y=start_y;i_y<end_y;i_y++)
                    {
                        for(int i_x=start_x;i_x<end_y;i_x++)
                        {
                            //Sum the product of each input channel with associated output channel filter
                            matrix_out[i][j]+=matrix_filter[i_y-start_y][i_x-start_x]*matrix_in[i_y][i_x];
                        }
                    }
                }
                //Add biases to each channel
                matrix_out[i][j]+=params->biases.v[c_out];
            }
        }
    }
    if(params->padding)
    {
        clear_tensor(p_input);
        free(p_input);
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
    double*** cube_filter = (double***)params->filters.v;
    double*** cube_filter_gradient = (double***) params->filters_gradients.v;
    double* biases_gradients= (double*)params->biases_gradients.v;
    #pragma omp parallel for
    for(int i=0;i<batch_size;i++)
    {
        tensor* gradient = &gradients[i];
        double*** cube_gradient = (double***)gradient->v;
        
        tensor* output = &layer->outputs[i];
        tensor p_gradient_previous;
        tensor* p_input = NULL;
        double*** cube_gradient_previous = (double***)layer->previous_gradients[i].v;
        if(params->padding)
        {
            p_input = pad_input(&layer->layer_inputs[i], layer);
            initialize_tensor(&p_gradient_previous, p_input->shape);
            cube_gradient_previous = (double***)p_gradient_previous.v;
        }
        const tensor* used_input = params->padding? p_input:&layer->layer_inputs[i];
        double*** cube_input = (double***)used_input->v;
        if (layer->activation)
        {
            //Back propagate the gradient error tensor
            gradient = layer->activation->activation_backward_propagation(&layer->activation_input[i], gradient, output, layer->activation);
        }
        for(int c_out=0;c_out<output_channels;c_out++)
        {
            double** matrix_filter = cube_filter[c_out];
            double** matrix_filter_gradient = cube_filter_gradient[c_out];
            double** matrix_gradient = cube_gradient[c_out];
            //Iterate trough each output height
            for(int i=0;i<output_height;i++)
            {
                //And trough each output width
                for(int j=0;j<output_width;j++)
                {
                    double gradient_ij = matrix_gradient[i][j];
                    //biases_gradient is the sum of batch gradients
                    biases_gradients[c_out]+=gradient_ij;
                    matrix_gradient[i][j]=0;
                    //Calculates the associated matrix slice locations in input space
                    int start_x=j*stride;
                    int start_y=i*stride;
                    int end_x=start_x+kernel_width;
                    int end_y=start_x+kernel_height;
                    //Iterate trough each input channel
                    for(int c_in=0;c_in<input_channels;c_in++)
                    {
                        double** matrix_in = cube_input[c_in];
                        double** matrix_gradient_previous= cube_gradient_previous[c_in];
                        //Iterate trough each cell of input slice
                        for(int i_y=start_y;i_y<end_y;i_y++)
                        {
                            for(int i_x=start_x;i_x<end_y;i_x++)
                            {
                                //Sum the product of each input channel with associated output gradient into the filter_gradient
                                matrix_filter_gradient[i_y-start_y][i_x-start_x]+=matrix_gradient[i][j]*matrix_in[i_y][i_x];
                                if(layer_index>0)
                                {
                                    matrix_gradient_previous[i_y][i_x]+=matrix_filter[i_y-start_y][i_x-start_x]*gradient_ij;
                                }                                
                            }
                        }
                    }
                }
            }
        }
        if(params->padding)
        {
            unpad_gradient_previous(&layer->previous_gradients[i], &p_gradient_previous, layer);
            clear_tensor(&p_gradient_previous);
            clear_tensor(p_input);
            free(p_input);
        }
    }
    layer->backward_calculation(optimizer, layer, layer_index);
    return layer->previous_gradients;
}

//Backward propagation function for Fully Connected layer (perceptron)
void backward_calculation_Conv2D(optimizer *optimizer, layer *layer, int layer_index)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    int output_channel = layer->output_shape->sizes[0];
    int output_height = layer->output_shape->sizes[1];
    int kernel_width = params->kernel_width;
    int kernel_height = params->kernel_height;
    double*** cube_filters = (double***)params->filters.v;
    double*** cube_filters_gradient = (double***)params->filters_gradients.v;
    double* biases_gradient = params->biases_gradients.v;
    int* iterator = get_iterator(&params->filters);
    //Update biases using new gradient
    for (int c_out = 0; c_out < output_channel; c_out++)
    {
        //Optimizer update bias
        params->biases.v[c_out] = optimizer->apply_gradient(params->biases.v[c_out], biases_gradient[c_out], layer_index, 0, &c_out, optimizer);
        //Reset gradient for next episode
        biases_gradient[c_out]=0;
        double** matrix_filters = cube_filters[c_out];
        double** matrix_filters_gradient = cube_filters_gradient[c_out];
        //Calculate the gradient for previous layer and update weights
        for(int i_y=0;i_y<kernel_height;i_y++)
        {
            for(int i_x=0;i_x<kernel_width;i_x++)
            {
                //Sum the product of each input channel with associated output channel filter
                optimizer->apply_gradient(matrix_filters[i_y][i_x], matrix_filters_gradient[i_y][i_x], layer_index, 1, iterator, optimizer);
                iterator = params->filters.get_next(&params->filters, iterator);
                //Reset gradient for next episode
                matrix_filters_gradient[i_y][i_x]=0;
            }
        }
    }
}

void init_memory_training_Conv2D(layer* layer)
{
    init_memory_training(layer);
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    initialize_tensor(&params->filters_gradients, params->filters.shape);
    initialize_tensor(&params->biases_gradients, params->biases.shape);
}

void clear_layer_training_memory_Conv2D(layer *layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    clear_layer_training_memory(layer);
    clear_tensor(&params->filters_gradients);
    clear_tensor(&params->biases_gradients);
}


void compile_layer_Conv2D(shape* input_shape, layer *layer)
{
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
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
    //Filter shape is created according to kernel_size and output_channel_size
    shape* filter_shape = build_shape(ThreeD);
    filter_shape->sizes[0]=params->n_output_channels;
    filter_shape->sizes[1]=params->kernel_height;
    filter_shape->sizes[2]=params->kernel_width;
    initialize_tensor(&params->filters, filter_shape);
    clear_shape(filter_shape);
    free(filter_shape);
    //Filters initialization
    double invert_rand_max = (double)1.0 / (double)RAND_MAX;
    // glorot uniform init: https://github.com/ElefHead/numpy-cnn/blob/master/utilities/initializers.py
    double limit = sqrt((double)6 /(input_shape->sizes[0] + (params->kernel_width*params->kernel_height*params->n_output_channels)));
    int* iterator = get_iterator(&params->filters);
    while(!params->filters.is_done(&params->filters, iterator))
    {
        params->filters.set_value(&params->filters, iterator, (2 * limit * ((double)rand() * invert_rand_max)) - limit);
        iterator = params->filters.get_next(&params->filters, iterator);
    }
    free(iterator);
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

void configure_default_layer(layer* layer)
{
    layer->forward_propagation_training_loop = forward_propagation_training_loop;
    layer->forward_propagation_predict_loop = forward_propagation_predict_loop;
    layer->init_predict_memory = init_memory_predict;
    layer->clear_predict_memory = clear_layer_predict_memory;
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
    layer->read_parameters = read_parameters_FC;
    layer->save_parameters = save_parameters_FC;
}

void configure_layer_Conv2D(layer* layer)
{
    configure_default_layer(layer);
    conv2D_parameters* params = (conv2D_parameters*)malloc(sizeof(conv2D_parameters));
    layer->parameters = params;
    layer->compile_layer = compile_layer_Conv2D;
    layer->init_training_memory = init_memory_training_Conv2D;
    layer->clear_training_memory = clear_layer_training_memory_Conv2D;
    layer->forward_calculation_training = forward_calculation_training_Conv2D;
    layer->forward_calculation_predict = forward_calculation_predict_Conv2D;
    layer->backward_calculation = backward_calculation_Conv2D;
    layer->build_shape_list = build_shape_list_Conv2D;
    layer->clear_parameters = clear_parameters_Conv2D;
    layer->read_parameters = read_parameters_Conv2D;
    layer->save_parameters = save_parameters_Conv2D;
}

layer* build_layer(layer_type type, shape* input_shape, shape* output_shape)
{
    layer *layer = (struct layer *)malloc(sizeof(struct layer));
    layer->type = type;
    layer->input_shape = input_shape;
    layer->output_shape = output_shape;
    switch(type)
    {
        default: configure_layer_FC(layer);break;
    }
}

layer *read_layer(FILE *fp)
{
    int type;
    int input_dimension, output_dimension;
    fscanf(fp, "input_shape:");
    shape* input_shape = read_shape(fp);
    fscanf(fp, ", output_shape:");
    shape* output_shape = read_shape(fp);
    fscanf(fp, ", type:%d\n", &type);
    if (type >= 0)
    {
        layer *layer = build_layer(type, input_shape, output_shape);
        layer->compile_layer(input_shape, layer);
        layer->read_parameters(fp, layer);
        layer->activation = read_activation(fp);
        return layer;
    }
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

layer* build_layer_Conv2D(int output_channel_size, int kernel_width, int kernel_height, int stride, short padding, activation* activation)
{
    layer *layer = (struct layer *)malloc(sizeof(struct layer));
    configure_layer_FC(layer);
    conv2D_parameters* params = (conv2D_parameters*)layer->parameters;
    params->n_output_channels = output_channel_size;
    params->kernel_width = kernel_width;
    params->kernel_height = kernel_height;
    params->stride = stride;
    params->padding = padding > 0;
    layer->type = CONV2D;
    layer->output_shape = build_shape(ThreeD);
    layer->output_shape->sizes[0]=output_channel_size;
    layer->activation = activation;
    return layer;
}
