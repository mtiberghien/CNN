#include "include/optimizer.h"
#include "math.h"
#include <stdlib.h>

//Clear the shape list
void clear_shape_list(shape_list* shape_list)
{
    for(int i=0;i<shape_list->n_shapes;i++)
    {
        clear_shape(&shape_list->shapes[i]);
    }
    if(shape_list->shapes)
    {
        free(shape_list->shapes);
    }
}
//Build an optimizer according to the provided type with default parameters
optimizer* build_optimizer(optimizer_type type)
{
    switch(type){
        case ADAM: return build_optimizer_Adam(1E-3,0.9,0.999,1E-7);
        default: return build_optimizer_GD(1E-2, 0);
    }
}
//Common compilation method
void compile_default(shape_list* layers_shape_list, int n_layers, struct optimizer* optimizer)
{
    optimizer->t=0;
    int total_params = 0;
    for(int i=0;i<n_layers;i++)
    {
        total_params+=layers_shape_list[i].n_shapes;
    }
    optimizer->n_parameters = total_params;
    optimizer->n_layers=n_layers;
}

//Save an optimizer to a file
void save_optimizer(FILE* fp, optimizer* optimizer)
{
    fprintf(fp, "n_layers:%d, n_parameters:%d, type:%d, t:%ld\n", optimizer->n_layers, optimizer->n_parameters, optimizer->type, optimizer->t);
    optimizer->save_params(fp, optimizer);
}

//Read an optimizer from a file
optimizer* read_optimizer(FILE* fp)
{
    int n_layers, n_parameters, type;
    long t;
    fscanf(fp, "n_layers:%d, n_parameters:%d, type:%d, t:%ld\n", &n_layers, &n_parameters, &type, &t);
    if(type>=0)
    {
        optimizer* optimizer = build_optimizer(type);
        optimizer->n_parameters = n_parameters;
        optimizer->n_layers = n_layers;
        optimizer->t = t;
        optimizer->read_params(fp, optimizer);
        return optimizer;
    }
    return NULL;
}