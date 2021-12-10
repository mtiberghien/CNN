#include "../../include/optimizer.h"
#include <stdlib.h>
//GD parameters structure
typedef struct gd_parameters{
    double alpha;
    double momentum;
    double velocity;
} gd_parameters;

//Clear GD memory
void clear_optimizer_gd(optimizer* optimizer)
{
    gd_parameters* params = (gd_parameters*)optimizer->parameters;
    free(params);
}

//Simple gradient descent calculation
double apply_gradient_GD(double value, double gradient, int layer_index, int param_index, int* tensor_indexes, optimizer* optimizer)
{
    gd_parameters* params = (gd_parameters*)optimizer->parameters;
    params->velocity = params->velocity*params->momentum + params->alpha*gradient;
    return value - params->velocity;
}
//Write GD optimizer parameters to file
void save_parameters_gd(FILE *fp, optimizer* optimizer)
{
    gd_parameters* params = (gd_parameters*)optimizer->parameters;
    fprintf(fp, "alpha:%le, momentum:%le, velocity:%le\n", params->alpha, params->momentum, params->velocity);
}

// Read GD optimizer paramaters from file
void read_parameters_gd(FILE *fp, optimizer* optimizer)
{
    gd_parameters* params = (gd_parameters*)optimizer->parameters;
    fscanf(fp, "alpha:%le, momentum:%le, velocity:%le\n", &params->alpha, &params->momentum, &params->velocity);;
}

//Build a simple gradient descent 
optimizer* build_optimizer_GD(double alpha, double momentum)
{
    //Memory allocation
    optimizer* result=(optimizer*) malloc(sizeof(optimizer));
    gd_parameters* params = malloc(sizeof(gd_parameters));
    //Store learning parameter
    params->alpha = alpha;
    params->momentum = momentum;
    params->velocity=0;
    result->parameters = params;
    //Set the gradient calculation functions
    result->apply_gradient=apply_gradient_GD;
    result->type = GD;
    result->compile = compile_default;
    result->clear = clear_optimizer_gd;
    result->save_params=save_parameters_gd;
    result->read_params=read_parameters_gd;
    return result;
}