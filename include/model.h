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
    void (*fit)(tensor* inputs, tensor* truths, int inputs_size, int batch_size, int epochs, struct model*);
    void (*compile)(optimizer* optimizer, loss*loss, struct model* model);
    optimizer* optimizer;
    loss* loss;
} model;

void compile(optimizer* optimizer, loss* loss, model* model);
void add_layer(layer* layer, model* model);
void remove_layer(int index, model* model);
tensor* predict(tensor* inputs, int inputs_size, model* model);
void fit(tensor* inputs, tensor* truths, int inputs_size, int batch_size, int epochs, model* model);
model* build_model();
void clear_model(model* model);

#endif