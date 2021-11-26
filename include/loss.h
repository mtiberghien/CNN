#ifndef LOSS_CNN
#define LOSS_CNN

#include "tensor.h"
#include <stdio.h>

typedef enum loss_type{CCE,MSE} loss_type;

typedef struct loss{
    loss_type type;
    double (*loss)(double truth, double output);
    double (*loss_prime)(double truth, double output);
    double (*forward_error_loop)(tensor* truths, tensor* outputs, int batch_size, double invert_output_size, struct loss* loss);
    tensor* (*backward_error_loop)(tensor* truths, tensor* outputs, int batch_size, double invert_output_size, struct loss* loss);
} loss;

double forward_error_loop(tensor* truths, tensor* outputs, int batch_size, double invert_output_size, loss* loss);
tensor* backward_error_loop(tensor* truths, tensor* outputs, int batch_size, double invert_output_size, loss* loss);

double loss_cce(double truth, double output);
double loss_prime_cce(double truth, double output);

double loss_mse(double truth, double output);
double loss_prime_mse(double truth, double output);

loss* build_loss(loss_type type);
loss* build_loss_cce();
loss* build_loss_mse();
void save_loss(FILE* fp, loss* loss);
loss* read_loss(FILE* fp);
#endif