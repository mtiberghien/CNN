#ifndef OPTIMIZER_CNN
#define OPTIMIZER_CNN

#include "tensor.h"

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
    void (*compile)(int* layers_output_size, int n_layers, struct optimizer* optimizer);
    void (*clear)(struct optimizer* optimizer);
} optimizer;

optimizer* build_optimizer(optimizer_type type);
optimizer* build_optimizer_GD(double alpha);
optimizer* build_optimizer_Adam(double alpha, double beta_1, double beta_2, double eps);

double apply_gradient_GD(double value, double gradient, int layer_index, int tensor_index, optimizer* optimizer);
double apply_gradient_Adam(double value, double gradient, int layer_index, int tensor_index, optimizer* optimizer);
void compile_default(int* layers_output_size, int n_layers, struct optimizer* optimizer);
void compile_Adam(int* layers_output_size, int n_layers, struct optimizer* optimizer);
void clear_optimizer_Adam(optimizer* optimizer);
void clear_optimizer_default(optimizer* optimizer);
#endif