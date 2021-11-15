#ifndef OPTIMIZER_CNN
#define OPTIMIZER_CNN

typedef struct optimizer{
    double alpha;
    double (*apply_gradient)(double value, double gradient);
} optimizer;


optimizer build_optimizer_SGD(double alpha);

double apply_gradient_SGD(double value, double gradient, optimizer optimizer);

#endif