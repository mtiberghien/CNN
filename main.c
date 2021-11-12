#include <stdio.h>

typedef struct operation{
    double (*execute)(double x, double y);
}operation;

double add(double x,double y);
double mult(double x, double y);
double add(double x, double y){
    double result = x+y;
    printf("%f+%f=%f\n", x,y,result);
    return result;
}

double mult(double x, double y){
    double result = x*y;
    printf("%f*%f=%f\n", x,y,result);
    return result;
}

int main(){
    operation op;
    op.execute = add;
    op.execute(3,5);
    op.execute = mult;
    op.execute(3,5);
}