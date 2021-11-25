#include "include/model.h"
#include "include/layer.h"
#include <stdlib.h>
#include <stdio.h>
#include "include/common.h"
#include <time.h>

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

tensor* predict(tensor* inputs, int inputs_size, model* model)
{
    tensor* outputs = inputs;
    double invert_inputs_size = (double)1.0/inputs_size;
    for(int i=0; i<model->n_layers;i++)
    {
        outputs = model->layers[i].forward_propagation_loop(outputs, inputs_size, invert_inputs_size, 0, &model->layers[i]);
    }
    return outputs;
}

training_result* fit(tensor* inputs, tensor* truths, int inputs_size, int batch_size, int epochs, model* model)
{
    //Random indices support
    int* indices = malloc(sizeof(int)*inputs_size);
    training_result* result = (training_result*)malloc(sizeof(training_result));
    int n_episodes = (inputs_size/batch_size + (inputs_size%batch_size == 0?0:1));
    result->n_results = n_episodes*epochs;
    result->loss = malloc(sizeof(double)*result->n_results);
    int result_indice=0;
    int mean_error_count = n_episodes/10;
    double invert_output_size = (double)1/model->layers[model->n_layers-1].output_size;
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
            double invert_batch_size = (double)1/current_batch_size;
            //Current batch Forward pass
            for(int i=0;i<model->n_layers;i++)
            {
                outputs = model->layers[i].forward_propagation_loop(outputs, current_batch_size, invert_batch_size, 1, &model->layers[i]);
            }
            //mean of errors of current batch
            loss = model->loss->forward_error_loop(truths_batch, outputs, current_batch_size, invert_batch_size, invert_output_size, model->loss);
            //Current batch Backward pass using mean of batch gradients
            tensor* mean_gradients = model->loss->backward_error_loop(truths_batch, outputs, current_batch_size, invert_batch_size, model->layers[model->n_layers-1].invert_output_size, model->loss);
            for(int i=model->n_layers-1;i>=0;i--)
            {
                mean_gradients = model->layers[i].backward_propagation(mean_gradients, model->optimizer, &model->layers[i], i);
            }
            clear_tensor(mean_gradients);
            free(mean_gradients);
            free(batch);
            free(truths_batch);
            remaining_size-=current_batch_size;
            trained+=current_batch_size;
            current_batch_size = min(batch_size, remaining_size);
            result->loss[result_indice]=loss;
            int start_indice = result_indice<mean_error_count-1?0:(result_indice - mean_error_count+1);
            double last_error_sum = 0;
            for(int i=start_indice;i<=result_indice;i++)
            {
                last_error_sum+=result->loss[i];
            }
            double last_mean_error=last_error_sum/(result_indice - start_indice +1);
            result_indice++;
            time(&step);
            printf("\033[2K\r%d/%d: %.2f%% - %.0fs - loss: %.4f",episode++,n_episodes, ((double)100*trained)/inputs_size, difftime(step,start), last_mean_error);
            fflush(stdout);
        }
        model->optimizer->t++;
        printf("\n");
    }
    free(indices);
    return result;
}

void compile(int input_size, optimizer* optimizer, loss* loss, model* model)
{
    model->loss=loss;
    model->optimizer=optimizer;
    int* layers_output_size = malloc(sizeof(int)*model->n_layers);
    for(int i=0;i<model->n_layers;i++)
    {
        model->layers[i].compile_layer(input_size, &model->layers[i]);
        input_size = model->layers[i].output_size;
        layers_output_size[i]=model->layers[i].output_size;
    }
    optimizer->compile(layers_output_size, model->n_layers, optimizer);
    free(layers_output_size);
}

model* build_model()
{
    model* result = (model*)malloc(sizeof(model));
    result->n_layers=0;
    result->add_layer=add_layer;
    result->remove_layer=remove_layer;
    result->predict=predict;
    result->fit=fit;
    result->compile = compile;
    return result;
}

void clear_model(model* model)
{
    for(int i=0;i<model->n_layers;i++)
    {
        clear_layer(&model->layers[i]);
    }
    model->n_layers=0;
    free(model->layers);
    model->optimizer->clear(model->optimizer);
    free(model->optimizer);
    free(model->loss);
    free(model);
}

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

void save_model(model* model, char* filename)
{
    FILE * fp;
    fp = fopen(filename, "w");
    if(fp != NULL)
    {
        fprintf(fp, "n_layers:%d\n", model->n_layers);
        for(int i=0;i<model->n_layers;i++)
        {
            save_layer(fp, &model->layers[i]);
        }
        save_optimizer(fp, model->optimizer);
        save_loss(fp, model->loss);
    }
    fclose(fp);
}

model* read_model(char* filename)
{
    model* model = build_model();
    FILE * fp;
    fp = fopen(filename, "r");
    int n_layers;
    if(fp!= NULL)
    {
        fscanf(fp, "n_layers:%d\n", &n_layers);
        for(int i=0;i<n_layers;i++)
        {
           layer* layer = read_layer(fp);
           if(layer!=NULL)
           {
               model->add_layer(layer, model);
           }
        }
        model->optimizer = read_optimizer(fp);
        model->loss = read_loss(fp);
    }
    return model;
    fclose(fp);
}