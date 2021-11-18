#ifndef LOSS_CNN
#define LOSS_CNNN

#include "tensor.h"

typedef struct loss{
    double (*loss)(tensor* truth, tensor* output);
    tensor* (*loss_prime)(tensor* truth, tensor* output, tensor* gradient);
    double (*forward_error_loop)(tensor* truths, tensor* outputs, int batch_size, struct loss* loss);
    tensor* (*backward_error_loop)(tensor* truths, tensor* outputs, int batch_size, struct loss* loss);
} loss;

double forward_error_loop(tensor* truths, tensor* outputs, int batch_size, loss* loss);
tensor* backward_error_loop(tensor* truths, tensor* outputs, int batch_size, loss* loss);

double loss_cce(tensor* truth, tensor* output);
tensor* loss_prime_cce(tensor* truth, tensor* output, tensor* gradient);

loss* build_loss_cce();
#endif