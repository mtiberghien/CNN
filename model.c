#include "include/model.h"
#include "include/layer.h"
#include <stdlib.h>
#include <stdio.h>
#include "include/common.h"

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

void fit(tensor* inputs, tensor* truths, int inputs_size, int batch_size, int epochs, model* model)
{
    //Random indices support
    int* indices = malloc(sizeof(int)*inputs_size);
    for(int i=0;i<inputs_size;i++)
    {
        indices[i]=i;
    }
    //Epoch loop
    for(int epoch=1;epoch<=epochs;epoch++)
    {
        printf("Epoch %d\n",epoch);
        int remaining_size = inputs_size;
        int current_batch_size = min(batch_size, remaining_size);
        double invert_batch_size = (double)1.0/current_batch_size;
        int main_indice=inputs_size-1;
        double mean_error =0;
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

            //Current batch Forward pass
            for(int i=0;i<model->n_layers;i++)
            {
                outputs = model->layers[i].forward_propagation_loop(outputs, current_batch_size, invert_batch_size, 1, &model->layers[i]);
            }
            //mean of errors of current batch
            mean_error = model->loss->forward_error_loop(truths_batch, outputs, current_batch_size, invert_batch_size, model->loss);

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
            current_batch_size = min(batch_size, remaining_size);
            invert_batch_size = batch_size <= remaining_size ? invert_batch_size : (double)1.0/current_batch_size;
        }
        printf("\tloss:%6.2f\n", mean_error);
    }
    free(indices);
}

void compile(optimizer* optimizer, loss* loss, model* model)
{
    model->loss=loss;
    model->optimizer=optimizer;
    int* layers_output_size = malloc(sizeof(int)*model->n_layers);
    for(int i=0;i<model->n_layers;i++)
    {
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