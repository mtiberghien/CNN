#ifndef CNN_TENSOR
#define CNN_TENSOR

//The data structure used in a neural network
typedef struct tensor{
    double* v;
    int size;
} tensor;

//Clear memory of a tensor
void clear_tensor(tensor);
//Clear memory of a collection of tensors
void clear_tensors(tensor*,int);

double sum(tensor*, double(*func)(double));

#endif

