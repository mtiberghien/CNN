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
        case ADAM: return build_optimizer_Adam(1E-3,0.9,0.999,1E-7);
        default: return build_optimizer_GD(1E-2);
    }
}

void compile_default(shape* layers_output_shapes, int n_layers, struct optimizer* optimizer)
{
    optimizer->t=0;
    optimizer->n_layers = n_layers;
}

//Simple gradient descent calculation
double apply_gradient_GD(double value, double gradient, int layer_index, int tensor_index, optimizer* optimizer)
{
    return value - (optimizer->alpha * gradient);
}

double apply_gradient_Adam(double value, double gradient, int layer_index, int tensor_index, optimizer* optimizer)
{
    optimizer->m[layer_index].v[tensor_index] = (optimizer->beta_1 * optimizer->m[layer_index].v[tensor_index]) + (1 - optimizer->beta_1) * gradient;
    optimizer->v[layer_index].v[tensor_index] = (optimizer->beta_2 * optimizer->v[layer_index].v[tensor_index]) + (1 - optimizer->beta_2) * pow(gradient,(double)2.0);
    double mhat = optimizer->m[layer_index].v[tensor_index]/(1 - pow(optimizer->beta_1, (double)optimizer->t+1));
    double vhat = optimizer->v[layer_index].v[tensor_index]/(1 - pow(optimizer->beta_2, (double)optimizer->t+1));
    return value - ((optimizer->alpha * mhat)/(sqrt(vhat)+optimizer->eps));
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

void compile_Adam(shape* layers_output_shapes, int n_layers, struct optimizer* optimizer)
{
    optimizer->n_layers = n_layers;
    optimizer->t=0;
    optimizer->m=(tensor*)malloc(sizeof(tensor)*n_layers);
    optimizer->v=(tensor*)malloc(sizeof(tensor)*n_layers);
    for(int i=0;i<n_layers;i++)
    {
        initialize_tensor(&optimizer->m[i], &layers_output_shapes[i]);
        initialize_tensor(&optimizer->v[i], &layers_output_shapes[i]);
    }
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

void save_optimizer(FILE* fp, optimizer* optimizer)
{
    fprintf(fp, "n_layers:%d, type:%d, alpha:%lf, beta_1:%lf, beta_2:%lf, eps:%lf\n", optimizer->n_layers, optimizer->type, optimizer->alpha, optimizer->beta_1, optimizer->beta_2, optimizer->eps);
    if(optimizer->type==ADAM)
    {
        for(int i=0;i<optimizer->n_layers;i++)
        {
            fprintf(fp, "shape:");
            save_shape(fp, optimizer->m[i].shape);
            fprintf(fp, "\n");
            save_tensor(fp, &optimizer->m[i]);
            save_tensor(fp, &optimizer->v[i]);
        }
    }
}

optimizer* read_optimizer(FILE* fp)
{
    int n_layers, type, size;
    double alpha, beta_1,beta_2,eps;
    fscanf(fp, "n_layers:%d, type:%d, alpha:%lf, beta_1:%lf, beta_2:%lf, eps:%lf\n", &n_layers, &type, &alpha, &beta_1, &beta_2, &eps);
    if(type>=0)
    {
        optimizer* optimizer = build_optimizer(type);
        optimizer->alpha=alpha;
        optimizer->beta_1=beta_1;
        optimizer->beta_2=beta_2;
        optimizer->eps=eps == 0 ? 1E-7:eps;
        optimizer->m=(tensor*)malloc(sizeof(tensor)*n_layers);
        optimizer->v=(tensor*)malloc(sizeof(tensor)*n_layers);
        optimizer->n_layers = n_layers;
        if(type==ADAM)
        {
            for(int i=0;i<n_layers;i++)
            {
                tensor* m = &optimizer->m[i];
                tensor* v = &optimizer->v[i];
                fscanf(fp, "shape:");
                shape* shape= read_shape(fp);
                fscanf(fp, "\n");
                initialize_tensor(m, shape);
                initialize_tensor(v, shape);
                read_tensor(fp, &optimizer->m[i]);
                read_tensor(fp, &optimizer->v[i]);
            }
        }
        return optimizer;
    }
    return NULL;
}