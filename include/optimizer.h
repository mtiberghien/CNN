#ifndef OPTIMIZER_CNN
#define OPTIMIZER_CNN

//Represents an optimization calculation
typedef struct optimizer{
    //learning rate
    double alpha;
    //gradient calculation
    double (*apply_gradient)(double value, double gradient, struct optimizer* optimizer);
} optimizer;


optimizer* build_optimizer_GD(double alpha);

double apply_gradient_GD(double value, double gradient, optimizer* optimizer);

#endif