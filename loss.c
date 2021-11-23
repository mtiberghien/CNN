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


tensor* backward_error_loop(tensor* truths, tensor* outputs, int batch_size, double invert_output_size, loss* loss)
{
    tensor* mean_gradients = (tensor*)malloc(sizeof(tensor));
    initialize_tensor(mean_gradients, outputs[0].size);
    for(int i=0;i<batch_size;i++)
    {
        loss->loss_prime(&truths[i], &outputs[i],mean_gradients, batch_size, invert_output_size);
    }
    return mean_gradients;
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
tensor* loss_prime_cce(tensor* truth, tensor* output, tensor* gradient, int batch_size, double invert_output_size)
{
    for(int i=0;i<output->size;i++)
    {
        double d = output->v[i]==0?1:output->v[i];
        gradient->v[i]-=(truth->v[i]/d);
    }
    return gradient;
}

double loss_mse(tensor* truth, tensor* output)
{
    double error = 0;
    for(int i=0;i<output->size;i++)
    {
        error+=pow(truth->v[i] - output->v[i], (double)2);
    }
    return error/output->size;
}
tensor* loss_prime_mse(tensor* truth, tensor* output, tensor* gradient, int batch_size, double invert_output_size)
{
    for(int i=0;i<output->size;i++)
    {
        double d = output->v[i]==0?1:output->v[i];
        gradient->v[i]+=2*(output->v[i] - truth->v[i])*invert_output_size;
    }
    return gradient;
}

loss* build_loss(loss_type type)
{
    switch(type){
        case CCE: return build_loss_cce();
        default: return build_loss_mse();
    }
}

loss* build_loss_cce()
{
    loss* result = (loss*)malloc(sizeof(loss));
    result->type = CCE;
    result->backward_error_loop = backward_error_loop;
    result->forward_error_loop= forward_error_loop;
    result->loss = loss_cce;
    result->loss_prime = loss_prime_cce;
}

loss* build_loss_mse()
{
    loss* result = (loss*)malloc(sizeof(loss));
    result->type = MSE;
    result->backward_error_loop = backward_error_loop;
    result->forward_error_loop= forward_error_loop;
    result->loss = loss_mse;
    result->loss_prime = loss_prime_mse;
}