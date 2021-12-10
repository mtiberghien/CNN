#include "include/model.h"
#include "include/layer.h"
#include <stdlib.h>
#include <stdio.h>
#include "include/common.h"
#include <time.h>
#include <string.h>
#include "include/utils.h"

//Add a layer to the model
void add_layer(layer* layer, model* model)
{
    if(model->n_layers==0)
    {
        model->layers=malloc(sizeof(struct layer));
    }
    else
    {
        model->layers= realloc(model->layers,(model->n_layers+1)*sizeof(struct layer));
    }
    model->layers[model->n_layers]=*layer;
    model->n_layers++;
}
//Remove a layer from the model using the provided index
void remove_layer(int index, model* model)
{
    if(index<model->n_layers && index>=0)
    {
        clear_layer(&model->layers[index]);
        for(int i=index;i<model->n_layers-1;i++)
        {
            model->layers[i]=model->layers[i+1];
        }
        model->layers=realloc(model->layers,(model->n_layers-1)*sizeof(layer));
        model->n_layers--;
    }
}
//Initialize the model memory required for predictions
void init_model_predict_memory(int batch_size, model* model)
{
    for(int i=0;i<model->n_layers;i++)
    {
        layer* layer = &model->layers[i];
        layer->batch_size = batch_size;
        layer->init_predict_memory(layer);
    }
}
//Clear the model memory required for predictions
void clear_model_predict_memory(model* model)
{
    for(int i=0;i<model->n_layers;i++)
    {
        model->layers[i].clear_predict_memory(&model->layers[i]);
    }
}
//Clear the memory of a trainin_result set
void clear_result(training_result* result)
{
    free(result->loss);
}
//Clear the memory of a trainin_result set. The training_result object is destroyed
void free_result(training_result* result)
{
    clear_result(result);
    free(result);
}
//Predict the outputs for a list of inputs
tensor* predict(tensor* inputs, int inputs_size, model* model)
{
    tensor* outputs = inputs;
    init_model_predict_memory(inputs_size, model);
    progression* progression = build_progression(inputs_size*model->n_layers, "predicting");
    for(int i=0; i<model->n_layers;i++)
    {
        outputs = model->layers[i].forward_propagation_predict_loop(outputs, inputs_size, &model->layers[i], progression);
    }
    free_progression(progression);
    return outputs;
}
//Initialize the model memory required for training
void init_model_training_memory(int batch_size, model* model)
{
    for(int i=0;i<model->n_layers;i++)
    {
        layer* layer = &model->layers[i];
        layer->batch_size = batch_size;
        layer->init_training_memory(layer);
    }
    layer last_layer = model->layers[model->n_layers-1];
    if(model->loss)
    {
        model->loss->init_training_memory(batch_size, last_layer.output_shape, model->loss);
    }
}
//Clear the model memory required for training
void clear_model_training_memory(model* model)
{
    for(int i=0;i<model->n_layers;i++)
    {
        model->layers[i].clear_training_memory(&model->layers[i]);
    }
    if(model->loss)
    {
        model->loss->clear_training_memory(model->loss);
    }
}
//Train the model using a list of input features and a list of truths, using random batchs and epochs
training_result* fit(tensor* inputs, tensor* truths, int inputs_size, int batch_size, int epochs, model* model)
{
    //Random indices support
    init_model_training_memory(batch_size, model);
    int* indices = (int*)malloc(sizeof(int)*inputs_size);
    training_result* result = (training_result*)malloc(sizeof(training_result));
    int n_episodes = (inputs_size/batch_size + (inputs_size%batch_size == 0?0:1));
    result->n_results = n_episodes*epochs;
    result->loss = malloc(sizeof(double)*result->n_results);
    int result_indice=0;
    int mean_error_count = n_episodes/10;
    mean_error_count = mean_error_count == 0 ?1:mean_error_count;
    for(int i=0;i<inputs_size;i++)
    {
        indices[i]=i;
    }
    time_t start,step;
    //Epoch loop
    for(int epoch=1;epoch<=epochs;epoch++)
    {
        printf("Epoch %d/%d\n",epoch, epochs);
        time(&start);
        int remaining_size = inputs_size;
        int current_batch_size = min(batch_size, remaining_size);
        int main_indice=inputs_size-1;
        double mean_error =0;
        double loss=0;
        int trained =0;
        int episode=1;
        //Execute all batches of an epoch
        while(current_batch_size>0)
        {
            //Initialize random batch without replace
            tensor* batch = (tensor*)malloc(sizeof(tensor)*current_batch_size);
            tensor* truths_batch = (tensor*)malloc(sizeof(tensor)*current_batch_size);
            #pragma omp parallel for
            for(int i=0;i<current_batch_size;i++)
            {
                int random_indice =((double)rand() / (double)RAND_MAX)*main_indice;
                int proposed_indice = indices[random_indice];
                batch[i]=inputs[proposed_indice];
                truths_batch[i]=truths[proposed_indice];
                indices[random_indice]=indices[main_indice];
                indices[main_indice]=main_indice;
                main_indice--;
            }
            tensor* outputs = batch;
            //Current batch Forward pass
            for(int i=0;i<model->n_layers;i++)
            {
                outputs = model->layers[i].forward_propagation_training_loop(outputs, current_batch_size, &model->layers[i], NULL);
            }
            //mean of errors of current batch
            loss = model->loss->forward_error_loop(truths_batch, outputs, current_batch_size, model->loss);
            //Current batch Backward pass using mean of batch gradients
            tensor* gradients = model->loss->backward_error_loop(truths_batch, outputs, current_batch_size, model->loss);
            for(int i=model->n_layers-1;i>=0;i--)
            {
                gradients = model->layers[i].backward_propagation_loop(gradients, model->optimizer, &model->layers[i], i);
            }
            free(batch);
            free(truths_batch);
            remaining_size-=current_batch_size;
            trained+=current_batch_size;
            current_batch_size = min(batch_size, remaining_size);
            result->loss[result_indice]=loss;
            int start_indice = result_indice<mean_error_count-1?0:(result_indice - mean_error_count+1);
            double last_error_sum = 0;
            #pragma omp parallel for reduction(+:last_error_sum)
            for(int i=start_indice;i<=result_indice;i++)
            {
                last_error_sum+=result->loss[i];
            }
            double last_mean_error=last_error_sum/(result_indice - start_indice +1);
            result_indice++;
            time(&step);
            printf("\033[K\r%d/%d: %.2f%% - %.0fs - loss: %.4f",episode++,n_episodes, ((double)100*trained)/inputs_size, difftime(step,start), last_mean_error);
            fflush(stdout);
        }
        model->optimizer->t++;
        printf("\n");
    }
    clear_model_training_memory(model);
    free(indices);
    return result;
}
//Compile the model, calculating the input and output for each layer
void compile(shape* input_shape, optimizer* optimizer, loss* loss, model* model)
{
    model->loss=loss;
    model->optimizer=optimizer;
    shape_list* layers_shape_list= malloc(sizeof(shape)*model->n_layers);
    for(int i=0;i<model->n_layers;i++)
    {
        model->layers[i].compile_layer(input_shape, &model->layers[i]);
        input_shape = model->layers[i].output_shape;
        model->layers[i].build_shape_list(&model->layers[i],&layers_shape_list[i]);
    }
    if(optimizer)
    {
        optimizer->compile(layers_shape_list, model->n_layers, optimizer);
    }
    for(int i=0;i<model->n_layers;i++)
    {
        clear_shape_list(&layers_shape_list[i]);
    }
    free(layers_shape_list);
}
//Write a separation line to the standard output in the summary context
void write_summary_line()
{
    printf("-----------------------------------------------------------\n");
}
//Print the shape to the standard output in the summary context
void summary_shape_to_string(char* shape_string, shape* shape)
{
    for(int i=0;i<shape->dimension;i++)
    {
        char size[20];
        sprintf(size, ",%d", shape->sizes[i]);
        strcat(shape_string,size);
    }
    strcat(shape_string, ")");
}
//Print the summary of a model to the standard output
void model_summary(model* model)
{
    write_summary_line();
    printf("%-20s%-20s%-20s\n","Id","Shape","Parameters");
    int total_parameters = 0;
    write_summary_line();
    for(int i=0;i<model->n_layers;i++)
    {
        
        layer* layer = &model->layers[i];
        char layer_name[20];
        int params_count = layer->get_trainable_parameters_count(layer);
        total_parameters+=params_count;
        char params[20];
        char shape_string[20]="(None";
        summary_shape_to_string(shape_string,layer->output_shape);
        sprintf(params, "%d",params_count);
        sprintf(layer_name, "%s_%d", layer->to_string(layer), i+1);
        printf("%-20s%-20s%-20s\n", layer_name,shape_string, params);
    }
    write_summary_line();
    printf("Total trainable parameters:%d\n", total_parameters);
}
//Initiliazes the memory and methods for a model
model* build_model()
{
    model* result = (model*)malloc(sizeof(model));
    result->n_layers=0;
    result->add_layer=add_layer;
    result->remove_layer=remove_layer;
    result->predict=predict;
    result->fit=fit;
    result->compile = compile;
    result->summary = model_summary;
    result->loss=NULL;
    result->optimizer=NULL;
    return result;
}
//Free the memory of a model
void clear_model(model* model)
{
    for(int i=0;i<model->n_layers;i++)
    {
        clear_layer(&model->layers[i]);
    }
    model->n_layers=0;
    free(model->layers);
    if(model->optimizer)
    {
        model->optimizer->clear(model->optimizer);
        free(model->optimizer);
    }
    if(model->loss)
    {
        free(model->loss);
    }
}
//Free the memory of a model. The model is destroyed
void free_model(model* model)
{
    clear_model(model);
    free(model);
}
//Write the training result to a file
void save_training_result(training_result* result, char* filename)
{
    FILE * fp;
    fp = fopen(filename, "w");
    if(fp != NULL)
    {
        fprintf(fp, "episode;loss\n");
        for(int i=0;i<result->n_results;i++)
        {
            fprintf(fp,"%d;%f\n", i,result->loss[i]);
        }
    }
    fclose(fp);
}
//Write a model to a file
void save_model(model* model, char* filename)
{
    FILE * fp;
    fp = fopen(filename, "w");
    if(fp != NULL)
    {
        fprintf(fp, "n_layers:%d\n", model->n_layers);
        for(int i=0;i<model->n_layers;i++)
        {
            fprintf(fp, "Layer %d\n", i+1);
            save_layer(fp, &model->layers[i]);
        }
        int optimizer_type = model->optimizer? model->optimizer->type :-1;
        fprintf(fp, "Optimizer:%d\n", optimizer_type);
        if(model->optimizer)
        {
            save_optimizer(fp, model->optimizer);
        }
        save_loss(fp, model->loss);
    }
    fclose(fp);
}
//Read a model from a file
model* read_model(char* filename)
{
    model* model = build_model();
    FILE * fp;
    fp = fopen(filename, "r");
    int n_layers;
    if(fp!= NULL)
    {
        fscanf(fp, "n_layers:%d\n", &n_layers);
        int layer_number;
        for(int i=0;i<n_layers;i++)
        {
           fscanf(fp, "Layer %d\n", &layer_number);
           layer* layer = read_layer(fp);
           if(layer!=NULL)
           {
               model->add_layer(layer, model);
           }
        }
        int optimizer_type;
        int loss_type;
        fscanf(fp, "Optimizer:%d\n",&optimizer_type);
        if(optimizer_type>=0)
        {
            model->optimizer = read_optimizer(fp);
        }
        model->loss = read_loss(fp);
        
    }
    return model;
    fclose(fp);
}