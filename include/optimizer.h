#ifndef OPTIMIZER_CNN
#define OPTIMIZER_CNN

#include "tensor.h"
#include <stdio.h>

typedef enum optimizer_type{GD, ADAM} optimizer_type;

//Represents an optimization calculation
typedef struct optimizer{
    optimizer_type type;
    int n_layers;
    long t;
    //learning rate
    double alpha;
    tensor* m;
    tensor* v;
    double beta_1;
    double beta_2;
    double eps;
    //gradient calculation
    double (*apply_gradient)(double value, double gradient, int layer_index, int tensor_index, struct optimizer* optimizer);
    void (*compile)(shape* layers_output_shapes, int n_layers, struct optimizer* optimizer);
    void (*clear)(struct optimizer* optimizer);
} optimizer;

optimizer* build_optimizer(optimizer_type type);
optimizer* build_optimizer_GD(double alpha);
optimizer* build_optimizer_Adam(double alpha, double beta_1, double beta_2, double eps);

void save_optimizer(FILE* fp, optimizer* optimizer);
optimizer* read_optimizer(FILE* fp);
#endif