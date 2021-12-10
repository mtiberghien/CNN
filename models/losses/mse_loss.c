#include "../../include/loss.h"
#include <math.h>
//MSE calculation
double loss_mse(double truth, double output)
{
    return pow(truth-output, (double)2);
}
//MSE derivative
double loss_prime_mse(double truth, double output)
{
    return 2*(output-truth);
}
//Build the Mean Squared Error loss 
loss* build_loss_mse()
{
    loss* result = create_default_loss(MSE);
    result->loss = loss_mse;
    result->loss_prime = loss_prime_mse;
}