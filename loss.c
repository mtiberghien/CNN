#include "include/tensor.h"
#include "include/loss.h"
#include "include/layer.h"
#include "math.h"

double forward_error_loop(tensor* truths, tensor* outputs,  int batch_size, double invert_output_size, loss* loss)
{
    double errors = 0;
    for(int i=0;i<batch_size;i++)
    {
        for(int j=0;j<outputs[0].size;j++)
        {
            errors+=invert_output_size*loss->loss(truths[i].v[j], outputs[i].v[j]);
        }
        
    }
    return errors/batch_size;
}


tensor* backward_error_loop(tensor* truths, tensor* outputs, int batch_size, double invert_output_size, loss* loss)
{
    for(int i=0;i<batch_size;i++)
    {
        tensor* gradient = &loss->gradients[i];
        for(int j=0;j<outputs[0].size;j++)
        {
            gradient->v[j]=invert_output_size*(loss->loss_prime(truths[i].v[j], outputs[i].v[j]));
        }
    }
    return loss->gradients;
}

double loss_cce(double truth, double output)
{
    return -truth*log(output);
}

double loss_prime_cce(double truth, double output)
{
    double d = output == 0?1:output;
    return -truth/output;
}

double loss_mse(double truth, double output)
{
    return pow(truth-output, (double)2);
}

double loss_prime_mse(double truth, double output)
{
    return 2*(output-truth);
}

void init_training_memory(int batch_size, int output_size, loss* loss)
{
    loss->batch_size=batch_size;
    loss->gradients = malloc(sizeof(tensor)*batch_size);
    for(int i=0;i<batch_size;i++)
    {
        initialize_tensor(&loss->gradients[i], output_size);
    }
}

void clear_training_memory(loss* loss)
{
    clear_tensors(loss->gradients, loss->batch_size);
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

loss* build_loss(loss_type type)
{
    switch(type){
        case CCE: return build_loss_cce();
        default: return build_loss_mse();
    }
}

void save_loss(FILE* fp, loss* loss)
{
    fprintf(fp, "loss:%d\n", loss->type);
}

loss* read_loss(FILE* fp)
{
    int type;
    fscanf(fp, "loss:%d\n", &type);
    if(type>=0)
    {
        return build_loss(type);
    }
    return NULL;
}