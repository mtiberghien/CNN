#ifndef MODEL_CNN
#define MODEL_CNN

#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "progression.h"


typedef struct training_result{
    int n_results;
    double* loss;
} training_result;

typedef struct model{
    int n_layers;
    struct layer* layers;
    void (*add_layer)(struct layer* layer, struct model*);
    void (*remove_layer)(int index, struct model*);
    tensor* (*predict)(tensor* inputs, int inputs_size, struct model*);
    training_result* (*fit)(tensor* inputs, tensor* truths, int inputs_size, int batch_size, int epochs, struct model*);
    void (*compile)(shape* shape, optimizer* optimizer, loss*loss, struct model* model);
    void (*summary)(struct model* model);
    optimizer* optimizer;
    loss* loss;
} model;

model* build_model();
void clear_model(model* model);
void save_training_result(training_result* result, char* filename);
void save_model(model* model, char* filename);
model* read_model(char* filename);
void clear_result(training_result* result);
void free_model(model* model);
void free_result(training_result* result);
#endif