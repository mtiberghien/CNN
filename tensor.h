#ifndef CNN_TENSOR
#define CNN_TENSOR

typedef struct tensor{
    double* t;
    int size;
} tensor;

void clear_tensor(tensor);

#endif
