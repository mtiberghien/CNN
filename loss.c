#include "include/tensor.h"
#include "include/loss.h"
#include "include/layer.h"
#include "math.h"

double forward_error_loop(tensor* truths, tensor* outputs,  int batch_size, loss* loss)
{
    double errors = 0;
    for(int i=0;i<batch_size;i++)
    {
        errors+=loss->loss(&truths[i], &outputs[i]);
    }
    return errors;
}


tensor* backward_error_loop(tensor* truths, tensor* outputs, int batch_size, loss* loss)
{
    tensor* gradients = (tensor*)malloc(batch_size*sizeof(tensor));
    for(int i=0;i<batch_size;i++)
    {
        tensor* gradient = &gradients[i];
        gradient->size = outputs[i].size;
        gradient->v=calloc(gradient->size, sizeof(double));
        loss->loss_prime(&truths[i], &outputs[i],gradient);
    }
}

double loss_cce(tensor* truth, tensor* output)
{
    double error = 0;
    for(int i=0;i<output->size;i++)
    {
        error-=truth->v[i]*log(output->v[i]);
    }
    return error;
}
tensor* loss_prime_cce(tensor* truth, tensor* output, tensor* gradient)
{
    for(int i=0;i<output->size;i++)
    {
        double d = output->v[i]==0?1:output->v[i];
        gradient->v[i]=-truth->v[i]/d;
    }
    return gradient;
}

loss* build_loss_cce()
{
    loss* result = (loss*)malloc(sizeof(loss));
    result->backward_error_loop = backward_error_loop;
    result->forward_error_loop= forward_error_loop;
    result->loss = loss_cce;
    result->loss_prime = loss_prime_cce;
}