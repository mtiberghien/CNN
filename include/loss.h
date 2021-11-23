#ifndef LOSS_CNN
#define LOSS_CNNN

#include "tensor.h"

typedef enum loss_type{CCE,MSE} loss_type;

typedef struct loss{
    loss_type type;
    double (*loss)(tensor* truth, tensor* output);
    tensor* (*loss_prime)(tensor* truth, tensor* output, tensor* gradient, int batch_size, double invert_output_size);
    double (*forward_error_loop)(tensor* truths, tensor* outputs, int batch_size, struct loss* loss);
    tensor* (*backward_error_loop)(tensor* truths, tensor* outputs, int batch_size, double invert_output_size, struct loss* loss);
} loss;

double forward_error_loop(tensor* truths, tensor* outputs, int batch_size, loss* loss);
tensor* backward_error_loop(tensor* truths, tensor* outputs, int batch_size, double invert_output_size, loss* loss);

double loss_cce(tensor* truth, tensor* output);
tensor* loss_prime_cce(tensor* truth, tensor* output, tensor* gradient, int batch_size, double invert_output_size);

double loss_mse(tensor* truth, tensor* output);
tensor* loss_prime_mse(tensor* truth, tensor* output, tensor* gradient, int batch_size, double invert_output_size);

loss* build_loss(loss_type type);
loss* build_loss_cce();
loss* build_loss_mse();
#endif