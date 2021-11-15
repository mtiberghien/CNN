#include "include/optimizer.h"

optimizer build_optimizer_SGD(double alpha)
{
    optimizer sgd;
    sgd.alpha = alpha;
    return sgd;
}

double apply_gradient_SGD(double value, double gradient, optimizer optimizer){
    return value - optimizer.alpha * gradient;
}