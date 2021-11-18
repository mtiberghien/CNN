#include "include/optimizer.h"
#include <stdlib.h>

//Build a simple gradient descent 
optimizer* build_optimizer_GD(double alpha)
{
    //Memory allocation
    optimizer* sgd=(optimizer*) malloc(sizeof(optimizer));
    //Store learning parameter
    sgd->alpha = alpha;
    //Set the gradient calculation function
    sgd->apply_gradient=apply_gradient_GD;
    return sgd;
}

//Simple gradient descent calculation
double apply_gradient_GD(double value, double gradient, optimizer* optimizer){
    return value - optimizer->alpha * gradient;
}