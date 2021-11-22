#ifndef MODEL_CNN
#define MODEL_CNNN

#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"


typedef struct training_result{
    int n_results;
    double* loss;
} training_result;

typedef struct model{
    int n_layers;
    layer* layers;
    void (*add_layer)(layer* layer, struct model*);
    void (*remove_layer)(int index, struct model*);
    tensor* (*predict)(tensor* inputs, int inputs_size, struct model*);
    training_result* (*fit)(tensor* inputs, tensor* truths, int inputs_size, int batch_size, int epochs, struct model*);
    void (*compile)(int input_size, optimizer* optimizer, loss*loss, struct model* model);
    optimizer* optimizer;
    loss* loss;
} model;

void compile(int input_size, optimizer* optimizer, loss* loss, model* model);
void add_layer(layer* layer, model* model);
void remove_layer(int index, model* model);
tensor* predict(tensor* inputs, int inputs_size, model* model);
training_result* fit(tensor* inputs, tensor* truths, int inputs_size, int batch_size, int epochs, model* model);
model* build_model();
void clear_model(model* model);
void save_training_result(training_result* result, char* filename);

#endif