#ifndef OPTIMIZER_CNN
#define OPTIMIZER_CNN

#include "tensor.h"
#include <stdio.h>

typedef enum optimizer_type{GD, ADAM} optimizer_type;

typedef struct shape_list{
    int n_shapes;
    shape* shapes;
} shape_list;

//Represents an optimization calculation
typedef struct optimizer{
    optimizer_type type;
    int n_parameters;
    int n_layers;
    long t;
    void* parameters;
    //gradient calculation
    double (*apply_gradient)(double value, double gradient, int layer_index, int param_index, int* tensor_indexes, struct optimizer* optimizer);
    void (*compile)(shape_list* layers_shape_list, int n_layers, struct optimizer* optimizer);
    void (*clear)(struct optimizer* optimizer);
    void (*save_params)(FILE*, struct optimizer*);
    void (*read_params)(FILE*, struct optimizer*);
    void (*increment_t)(struct optimizer*);
} optimizer;

extern optimizer* build_optimizer(optimizer_type type);
extern optimizer* build_optimizer_GD(double alpha, double momentum);
extern optimizer* build_optimizer_Adam(double alpha, double beta_1, double beta_2, double eps);
optimizer* build_default_optimizer();
void save_optimizer(FILE* fp, optimizer* optimizer);
optimizer* read_optimizer(FILE* fp);
void clear_shape_list(shape_list*);
void compile_default(shape_list* layers_shape_list, int n_layers, struct optimizer* optimizer);
void optimizer_increment_t(optimizer* optimizer);
#endif