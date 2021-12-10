#include "../../include/optimizer.h"
#include "../../include/tensor.h"
#include <math.h>
#include <stdlib.h>

typedef struct adam_parameters{
    //learning rate
    double alpha;
    tensor* m;
    tensor* v;
    double beta_1;
    double beta_2;
    double eps;
    int* indexes;
} adam_parameters;

void clear_optimizer_adam(optimizer* optimizer)
{
    adam_parameters* params = (adam_parameters*)optimizer->parameters;
    free_tensors(params->m, optimizer->n_parameters);
    free_tensors(params->v, optimizer->n_parameters);
    free(params->indexes);
    free(params);
}

double apply_gradient_Adam(double value, double gradient, int layer_index, int param_index, int* tensor_indexes, optimizer* optimizer)
{
    adam_parameters* params = (adam_parameters*)optimizer->parameters;
    int index = params->indexes[layer_index];
    tensor* m = &params->m[index+param_index];
    tensor* v = &params->v[index+param_index];
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
    fprintf(fp, "alpha:%le, beta_1:%le, beta_2:%le, eps:%le\n", params->alpha, params->beta_1, params->beta_2, params->eps);
    fprintf(fp, "indexes:");
    for(int i=0;i<optimizer->n_layers;i++)
    {
        fprintf(fp, "%d", params->indexes[i]);
        if(i<optimizer->n_layers-1)
        {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "\n");
    for(int i=0;i<optimizer->n_parameters;i++)
    {
        fprintf(fp, "shape:");
        save_shape(fp, params->m[i].shape);
        fprintf(fp, "\n");
        save_tensor(fp, &params->m[i]);
        save_tensor(fp, &params->v[i]);
    }
}

void read_parameters_adam(FILE *fp, optimizer* optimizer)
{
    adam_parameters* params = (adam_parameters*)optimizer->parameters;
    int n_parameters = optimizer->n_parameters;
    fscanf(fp, "alpha:%le, beta_1:%le, beta_2:%le, eps:%le\n", &params->alpha, &params->beta_1, &params->beta_2, &params->eps);
    params->m=(tensor*)malloc(sizeof(tensor)*n_parameters);
    params->v=(tensor*)malloc(sizeof(tensor)*n_parameters);
    params->indexes = (int*)malloc(sizeof(int)*optimizer->n_layers);
    fscanf(fp, "indexes:");
    for(int i=0;i<optimizer->n_layers;i++)
    {
        fscanf(fp, "%d", &params->indexes[i]);
        if(i<optimizer->n_layers-1)
        {
            fscanf(fp, ",");
        }
    }
    fscanf(fp, "\n");
    for(int i=0;i<n_parameters;i++)
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

void compile_Adam(shape_list* layers_shape_list, int n_layers, struct optimizer* optimizer)
{
    compile_default(layers_shape_list, n_layers, optimizer);
    int total_params = optimizer->n_parameters;
    adam_parameters* params =(adam_parameters*)optimizer->parameters;
    params->m=(tensor*)malloc(sizeof(tensor)*total_params);
    params->v=(tensor*)malloc(sizeof(tensor)*total_params);
    params->indexes=(int*)malloc(sizeof(int)*n_layers);
    int index =0;
    for(int i=0;i<n_layers;i++)
    {
        params->indexes[i]=index;
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