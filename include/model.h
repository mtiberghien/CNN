#ifndef MODEL_CNN
#define MODEL_CNNN

#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"

typedef struct model{
    int n_layers;
    layer* layers;
    void (*add_layer)(layer* layer, struct model*);
    void (*remove_layer)(int index, struct model*);
    tensor* (*predict)(tensor* inputs, int inputs_size, struct model*);
    void (*fit)(tensor* inputs, tensor* truths, int inputs_size, int batch_size, struct model*);
    optimizer* optimizer;
    loss* loss;
} model;


void add_layer(layer* layer, model* model);
void remove_layer(int index, model* model);
tensor* predict(tensor* inputs, int inputs_size, model* model);
void fit(tensor* inputs, tensor* truths, int inputs_size, int batch_size, model* model);


#endif