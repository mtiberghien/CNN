#ifndef LOSS_CNN
#define LOSS_CNN

#include "tensor.h"
#include <stdio.h>

typedef enum loss_type{CCE,MSE} loss_type;

typedef struct loss{
    loss_type type;
    int batch_size;
    tensor* gradients;
    void (*init_training_memory)(int batch_size, shape* shape, struct loss*);
    void (*clear_training_memory)(struct loss*);
    double (*loss)(double truth, double output);
    double (*loss_prime)(double truth, double output);
    double (*forward_error_loop)(tensor* truths, tensor* outputs, int batch_size, struct loss* loss);
    tensor* (*backward_error_loop)(tensor* truths, tensor* outputs, int batch_size, struct loss* loss);
} loss;


extern loss* build_loss(loss_type type);
void save_loss(FILE* fp, loss* loss);
loss* read_loss(FILE* fp);
loss* build_loss_cce();
loss* build_loss_mse();
loss* create_default_loss(loss_type type);
#endif