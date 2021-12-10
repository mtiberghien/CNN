#include "../../include/loss.h"
#include <math.h>

//CCE calculation
double loss_cce(double truth, double output)
{
    double result = -truth*log(output);
    return result;
}

//CCE derivative
double loss_prime_cce(double truth, double output)
{
    double d = output == 0?1:output;
    return -truth/output;
}

//Build categorical cross entropy loss
loss* build_loss_cce()
{
    loss* result = create_default_loss(CCE);
    result->loss = loss_cce;
    result->loss_prime = loss_prime_cce;
}