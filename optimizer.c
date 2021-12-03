#include "include/optimizer.h"
#include "math.h"
#include <stdlib.h>

typedef struct adam_parameters{
    //learning rate
    double alpha;
    tensor* m;
    tensor* v;
    double beta_1;
    double beta_2;
    double eps;
} adam_parameters;

typedef struct gd_parameters{
    double alpha;
} gd_parameters;

void clear_optimizer_adam(optimizer* optimizer)
{
    adam_parameters* params = (adam_parameters*)optimizer->parameters;
    clear_tensors(params->m, optimizer->n_layers);
    clear_tensors(params->v, optimizer->n_layers);
    free(params->m);
    free(params->v);
    free(params);
}

void clear_optimizer_gd(optimizer* optimizer)
{
    gd_parameters* params = (gd_parameters*)optimizer->parameters;
    free(params);
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
    double alpha = ((gd_parameters*)optimizer->parameters)->alpha;
    return value - (alpha * gradient);
}

double apply_gradient_Adam(double value, double gradient, int layer_index, int tensor_index, optimizer* optimizer)
{
    adam_parameters* params = (adam_parameters*)optimizer->parameters;
    params->m[layer_index].v[tensor_index] = (params->beta_1 * params->m[layer_index].v[tensor_index]) + (1 - params->beta_1) * gradient;
    params->v[layer_index].v[tensor_index] = (params->beta_2 * params->v[layer_index].v[tensor_index]) + (1 - params->beta_2) * pow(gradient,(double)2.0);
    double mhat = params->m[layer_index].v[tensor_index]/(1 - pow(params->beta_1, (double)optimizer->t+1));
    double vhat = params->v[layer_index].v[tensor_index]/(1 - pow(params->beta_2, (double)optimizer->t+1));
    return value - ((params->alpha * mhat)/(sqrt(vhat)+params->eps));
}

void save_parameters_adam(FILE *fp, optimizer* optimizer)
{
    adam_parameters* params = (adam_parameters*)optimizer->parameters;
    fprintf(fp, "alpha:%lf, beta_1:%lf, beta_2:%lf, eps:%lf\n", params->alpha, params->beta_1, params->beta_2, params->eps);
    for(int i=0;i<optimizer->n_layers;i++)
    {
        fprintf(fp, "shape:");
        save_shape(fp, params->m[i].shape);
        fprintf(fp, "\n");
        save_tensor(fp, &params->m[i]);
        save_tensor(fp, &params->v[i]);
    }
}

void save_parameters_gd(FILE *fp, optimizer* optimizer)
{
    gd_parameters* params = (gd_parameters*)optimizer->parameters;
    fprintf(fp, "alpha:%lf\n", params->alpha);
}

void read_parameters_adam(FILE *fp, optimizer* optimizer)
{
    adam_parameters* params = (adam_parameters*)optimizer->parameters;
    int n_layers = optimizer->n_layers;
    fscanf(fp, "alpha:%lf, beta_1:%lf, beta_2:%lf, eps:%lf\n", &params->alpha, &params->beta_1, &params->beta_2, &params->eps);
    params->eps=params->eps == 0 ? 1E-7:params->eps;
    params->m=(tensor*)malloc(sizeof(tensor)*n_layers);
    params->v=(tensor*)malloc(sizeof(tensor)*n_layers);
    for(int i=0;i<n_layers;i++)
    {
        tensor* m = &params->m[i];
        tensor* v = &params->v[i];
        fscanf(fp, "shape:");
        shape* shape= read_shape(fp);
        fscanf(fp, "\n");
        initialize_tensor(m, shape);
        initialize_tensor(v, shape);
        read_tensor(fp, &params->m[i]);
        read_tensor(fp, &params->v[i]);
    }
}

void read_parameters_gd(FILE *fp, optimizer* optimizer)
{
    gd_parameters* params = (gd_parameters*)optimizer->parameters;
    fscanf(fp, "alpha:%lf\n", &params->alpha);
}

//Build a simple gradient descent 
optimizer* build_optimizer_GD(double alpha)
{
    //Memory allocation
    optimizer* result=(optimizer*) malloc(sizeof(optimizer));
    gd_parameters* params = malloc(sizeof(gd_parameters));
    //Store learning parameter
    params->alpha = alpha;
    result->parameters = params;
    //Set the gradient calculation function
    result->apply_gradient=apply_gradient_GD;
    result->type = GD;
    result->compile = compile_default;
    result->clear = clear_optimizer_gd;
    result->save_params=save_parameters_gd;
    result->read_params=read_parameters_gd;
    return result;
}

void compile_Adam(shape* layers_output_shapes, int n_layers, struct optimizer* optimizer)
{
    optimizer->n_layers = n_layers;
    optimizer->t=0;
    adam_parameters* params =(adam_parameters*)optimizer->parameters;
    params->m=(tensor*)malloc(sizeof(tensor)*n_layers);
    params->v=(tensor*)malloc(sizeof(tensor)*n_layers);
    for(int i=0;i<n_layers;i++)
    {
        initialize_tensor(&params->m[i], &layers_output_shapes[i]);
        initialize_tensor(&params->v[i], &layers_output_shapes[i]);
    }
}

optimizer* build_optimizer_Adam(double alpha, double beta_1, double beta_2, double eps)
{
    optimizer* result = (optimizer*) malloc(sizeof(optimizer));
    adam_parameters* params = malloc(sizeof(adam_parameters));
    params->alpha = alpha;
    params->beta_1 = beta_1;
    params->beta_2 = beta_2;
    params->eps = eps;
    result->type = ADAM;
    result->parameters = params;
    result->compile = compile_Adam;
    result->apply_gradient= apply_gradient_Adam;
    result->clear=clear_optimizer_adam;
    result->save_params=save_parameters_adam;
    result->read_params=read_parameters_adam;
    return result;
}

void save_optimizer(FILE* fp, optimizer* optimizer)
{
    fprintf(fp, "n_layers:%d, type:%d, t:%ld\n", optimizer->n_layers, optimizer->type, optimizer->t);
    optimizer->save_params(fp, optimizer);
}

optimizer* read_optimizer(FILE* fp)
{
    int n_layers, type;
    long t;
    fscanf(fp, "n_layers:%d, type:%d, t:%ld\n", &n_layers, &type, &t);
    if(type>=0)
    {
        optimizer* optimizer = build_optimizer(type);
        optimizer->n_layers = n_layers;
        optimizer->t = t;
        optimizer->read_params(fp, optimizer);
        return optimizer;
    }
    return NULL;
}