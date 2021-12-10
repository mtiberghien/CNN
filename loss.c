#include "include/tensor.h"
#include "include/loss.h"
#include "include/layer.h"
#include "math.h"

double forward_error_loop(tensor* truths, tensor* outputs,  int batch_size, loss* loss)
{
    double errors = 0;
    int* size = loss->gradients[0].shape->sizes;
    int dim = loss->gradients[0].shape->dimension;
    double tot_size =1;
    for(int i=0;i<dim;i++)
    {
        tot_size*=size[i];
    }
    double invert_output_size = 1/tot_size;
    #pragma omp parallel for reduction(+:errors)
    for(int i=0;i<batch_size;i++)
    {
        tensor* truth = &truths[i];
        tensor* output = &outputs[i];
        int* iterator = get_iterator(truth);
        while(!truth->is_done(truth, iterator))
        {
            double truth_v = truth->get_value(truth, iterator);
            double output_v = output->get_value(output, iterator);
            errors+=invert_output_size*loss->loss(truth_v, output_v);
            iterator = truth->get_next(truth, iterator);
        }
        free(iterator);     
    }
    return errors/batch_size;
}


tensor* backward_error_loop(tensor* truths, tensor* outputs, int batch_size, loss* loss)
{
    int* size = loss->gradients[0].shape->sizes;
    int dim = loss->gradients[0].shape->dimension;
    double tot_size =1;
    for(int i=0;i<dim;i++)
    {
        tot_size*=size[i];
    }
    double invert_output_size = 1/tot_size;
    #pragma omp parallel for
    for(int i=0;i<batch_size;i++)
    {
        tensor* truth = &truths[i];
        tensor* output = &outputs[i];
        tensor* gradient = &loss->gradients[i];
        int* iterator = get_iterator(truth);
        while(!truth->is_done(truth, iterator))
        {
            double truth_v = truth->get_value(truth, iterator);
            double output_v = output->get_value(output, iterator);
            gradient->set_value(gradient, iterator, invert_output_size*(loss->loss_prime(truth_v, output_v)));
            iterator = truth->get_next(truth, iterator);
        }
        free(iterator); 
    }
    return loss->gradients;
}

void init_training_memory(int batch_size, shape* shape, loss* loss)
{
    loss->batch_size=batch_size;
    loss->gradients = malloc(sizeof(tensor)*batch_size);
    for(int i=0;i<batch_size;i++)
    {
        initialize_tensor(&loss->gradients[i], shape);
    }
}

void clear_training_memory(loss* loss)
{
    clear_tensors(loss->gradients, loss->batch_size);
}

loss* create_default_loss(loss_type type)
{
    loss* result = (loss*)malloc(sizeof(loss));
    result->type = type;
    result->backward_error_loop = backward_error_loop;
    result->forward_error_loop= forward_error_loop;
    result->backward_error_loop = backward_error_loop;
    result->forward_error_loop= forward_error_loop;
    result->init_training_memory=init_training_memory;
    result->clear_training_memory=clear_training_memory;
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
    int type = loss?loss->type:-1;
    fprintf(fp, "Loss:%d\n", type);
}

loss* read_loss(FILE* fp)
{
    int type;
    fscanf(fp, "Loss:%d\n", &type);
    if(type>=0)
    {
        return build_loss(type);
    }
    return NULL;
}