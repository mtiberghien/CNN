#include "include/model.h"
#include <stdlib.h>

void add_layer(layer* layer, model* model)
{
    model->layers= realloc(model->layers,(model->n_layers+1)*sizeof(layer));
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
tensor* predict(tensor* inputs, int inputs_size, model* model);
void fit(tensor* inputs, tensor* truths, int inputs_size, int batch_size, model* model);