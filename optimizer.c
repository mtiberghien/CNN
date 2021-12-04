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
    clear_tensors(params->m, optimizer->n_parameters);
    clear_tensors(params->v, optimizer->n_parameters);
    free(params->m);
    free(params->v);
    free(params);
}

void clear_optimizer_gd(optimizer* optimizer)
{
    gd_parameters* params = (gd_parameters*)optimizer->parameters;
    free(params);
}

void clear_shape_list(shape_list* shape_list)
{
    for(int i=0;i<shape_list->n_shapes;i++)
    {
        clear_shape(&shape_list->shapes[i]);
    }
    free(shape_list->shapes);
}

optimizer* build_optimizer(optimizer_type type)
{
    switch(type){
        case ADAM: return build_optimizer_Adam(1E-3,0.9,0.999,1E-7);
        default: return build_optimizer_GD(1E-2);
    }
}

void compile_default(shape_list* layers_shape_list, int n_layers, struct optimizer* optimizer)
{
    optimizer->t=0;
    int total_params = 0;
    for(int i=0;i<n_layers;i++)
    {
        total_params+=layers_shape_list[i].n_shapes;
    }
    optimizer->n_parameters = total_params;
}

//Simple gradient descent calculation
double apply_gradient_GD(double value, double gradient, int layer_index, int param_index, int* tensor_indexes, optimizer* optimizer)
{
    double alpha = ((gd_parameters*)optimizer->parameters)->alpha;
    return value - (alpha * gradient);
}

double apply_gradient_Adam(double value, double gradient, int layer_index, int param_index, int* tensor_indexes, optimizer* optimizer)
{
    adam_parameters* params = (adam_parameters*)optimizer->parameters;
    tensor* m = &params->m[layer_index+param_index];
    tensor* v = &params->v[layer_index+param_index];
    double m_value = m->get_value(m, tensor_indexes);
    double m_value_next = (params->beta_1 * m_value) + (1 - params->beta_1) * gradient;
    double v_value = v->get_value(v, tensor_indexes);
    double v_value_next = (params->beta_2 * v_value) + (1 - params->beta_2) * pow(gradient,(double)2.0);
    double mhat = m_value_next/(1 - pow(params->beta_1, (double)optimizer->t+1));
    double vhat = v_value_next/(1 - pow(params->beta_2, (double)optimizer->t+1));
    m->set_value(m, tensor_indexes, m_value_next);
    v->set_value(v, tensor_indexes, v_value_next);
    return value - ((params->alpha * mhat)/(sqrt(vhat)+params->eps));
}

void save_parameters_adam(FILE *fp, optimizer* optimizer)
{
    adam_parameters* params = (adam_parameters*)optimizer->parameters;
    fprintf(fp, "alpha:%lf, beta_1:%lf, beta_2:%lf, eps:%lf\n", params->alpha, params->beta_1, params->beta_2, params->eps);
    for(int i=0;i<optimizer->n_parameters;i++)
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
    int n_layers = optimizer->n_parameters;
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

void compile_Adam(shape_list* layers_shape_list, int n_layers, struct optimizer* optimizer)
{
    compile_default(layers_shape_list, n_layers, optimizer);
    int total_params = optimizer->n_parameters;
    adam_parameters* params =(adam_parameters*)optimizer->parameters;
    params->m=(tensor*)malloc(sizeof(tensor)*total_params);
    params->v=(tensor*)malloc(sizeof(tensor)*total_params);
    int index =0;
    for(int i=0;i<n_layers;i++)
    {
        shape_list s_l = layers_shape_list[i];
        for(int j=0;j<s_l.n_shapes;j++)
        {
            initialize_tensor(&params->m[index], &s_l.shapes[j]);
            initialize_tensor(&params->v[index], &s_l.shapes[j]);
            index++;
        }
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
    fprintf(fp, "n_parameters:%d, type:%d, t:%ld\n", optimizer->n_parameters, optimizer->type, optimizer->t);
    optimizer->save_params(fp, optimizer);
}

optimizer* read_optimizer(FILE* fp)
{
    int n_layers, type;
    long t;
    fscanf(fp, "n_parameters:%d, type:%d, t:%ld\n", &n_layers, &type, &t);
    if(type>=0)
    {
        optimizer* optimizer = build_optimizer(type);
        optimizer->n_parameters = n_layers;
        optimizer->t = t;
        optimizer->read_params(fp, optimizer);
        return optimizer;
    }
    return NULL;
}