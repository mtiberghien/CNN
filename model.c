#include "include/model.h"
#include "include/layer.h"
#include <stdlib.h>

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
    for(int i=0; i<model->n_layers;i++)
    {
        outputs = model->layers[i].forward_propagation_loop(outputs, inputs_size, 0, &model->layers[i]);
    }
    return outputs;
}
void fit(tensor* inputs, tensor* truths, int inputs_size, int batch_size, model* model)
{

}

model* build_model(loss* loss, optimizer* optimizer)
{
    model* result = (model*)malloc(sizeof(model));
    result->n_layers=0;
    result->add_layer=add_layer;
    result->remove_layer=remove_layer;
    result->predict=predict;
    result->fit=fit;
    result->loss=loss;
    result->optimizer=optimizer;
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
    free(model->optimizer);
    free(model->loss);
    free(model);
}