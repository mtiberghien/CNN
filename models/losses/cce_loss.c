#include "../../include/loss.h"
#include <math.h>

double loss_cce(double truth, double output)
{
    double result = -truth*log(output);
    return result;
}

double loss_prime_cce(double truth, double output)
{
    double d = output == 0?1:output;
    return -truth/output;
}

loss* build_loss_cce()
{
    loss* result = create_default_loss(CCE);
    result->loss = loss_cce;
    result->loss_prime = loss_prime_cce;
}