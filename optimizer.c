#include "include/optimizer.h"
#include "math.h"
#include <stdlib.h>

void clear_optimizer_Adam(optimizer* optimizer)
{
    clear_tensors(optimizer->m, optimizer->n_layers);
    clear_tensors(optimizer->v, optimizer->n_layers);
    free(optimizer->m);
    free(optimizer->v);
}

void clear_optimizer_default(optimizer* optimizer)
{
}

optimizer* build_optimizer(optimizer_type type)
{
    switch(type){
        case GD: 
        case ADAM: return build_optimizer_Adam(1E-3,0.9,0.999,1E-7);
        default: return build_optimizer_GD(1E-3);
    }
}

//Build a simple gradient descent 
optimizer* build_optimizer_GD(double alpha)
{
    //Memory allocation
    optimizer* result=(optimizer*) malloc(sizeof(optimizer));
    //Store learning parameter
    result->alpha = alpha;
    //Set the gradient calculation function
    result->apply_gradient=apply_gradient_GD;
    result->type = GD;
    result->compile = compile_default;
    result->clear = clear_optimizer_default;
    return result;
}

optimizer* build_optimizer_Adam(double alpha, double beta_1, double beta_2, double eps)
{
    optimizer* result = (optimizer*) malloc(sizeof(optimizer));
    result->alpha = alpha;
    result->beta_1 = beta_1;
    result->beta_2 = beta_2;
    result->eps = eps;
    result->type = ADAM;
    result->compile = compile_Adam;
    result->apply_gradient= apply_gradient_Adam;
    result->clear=clear_optimizer_Adam;
    return result;
}

//Simple gradient descent calculation
double apply_gradient_GD(double value, double gradient, int layer_index, int tensor_index, optimizer* optimizer)
{
    return value - (optimizer->alpha * gradient);
}

double apply_gradient_Adam(double value, double gradient, int layer_index, int tensor_index, optimizer* optimizer)
{
    (&optimizer->m[layer_index])->v[tensor_index] = (optimizer->beta_1 * optimizer->m[layer_index].v[tensor_index]) + (1 - optimizer->beta_1) * gradient;
    (&optimizer->v[layer_index])->v[tensor_index] = (optimizer->beta_2 * optimizer->v[layer_index].v[tensor_index]) + (1 - optimizer->beta_2) * pow(gradient,(double)2.0);
    double mhat = optimizer->m[layer_index].v[tensor_index]/(1 - pow(optimizer->beta_1, (double)optimizer->t+1));
    double vhat = optimizer->v[layer_index].v[tensor_index]/(1 - pow(optimizer->beta_2, (double)optimizer->t+1));
    return value - ((optimizer->alpha * mhat)/(sqrt(vhat)+optimizer->eps));
}

void compile_default(int* layers_output_size, int n_layers, struct optimizer* optimizer)
{
    optimizer->t=0;
    optimizer->n_layers = n_layers;
}
void compile_Adam(int* layers_output_size, int n_layers, struct optimizer* optimizer)
{
    optimizer->n_layers = n_layers;
    optimizer->t=0;
    optimizer->m=(tensor*)malloc(sizeof(tensor)*n_layers);
    optimizer->v=(tensor*)malloc(sizeof(tensor)*n_layers);
    for(int i=0;i<n_layers;i++)
    {
        initialize_tensor(&optimizer->m[i], layers_output_size[i]);
        initialize_tensor(&optimizer->v[i], layers_output_size[i]);
    }
}