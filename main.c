#include <stdio.h>
#include <stdlib.h>
#include "include/model.h"
#include "math.h"
#include "include/common.h"
#include "time.h"

int main(){
    model* model = build_model();
    int inputs_size = 11;
    tensor* x = malloc(sizeof(tensor)*inputs_size);
    tensor* y = malloc(sizeof(tensor)*inputs_size);
    for(int i=0;i<inputs_size;i++)
    {
        initialize_tensor(&x[i],1);
        x[i].v[0]=(i-5)*5;
        initialize_tensor(&y[i],1);
        y[i].v[0]=32+(1.8*x[i].v[0]);
    }

    model->add_layer(build_layer_FC(1,1, NULL), model);
    model->compile(build_optimizer_GD(1E-3), build_loss_mse(), model);
    training_result* result = model->fit(x, y, inputs_size, 1, 370, model);

    tensor x_pred;
    initialize_tensor(&x_pred,1);
    x_pred.v[0]=100;
    tensor* y_pred = model->predict(&x_pred,1, model);
    printf("%6.2f °C -> %6.2f °F\n", x_pred.v[0], y_pred->v[0]);
    printf("weight:%6.2f, bias:%6.2f\n", model->layers[0].weights[0].v[0], model->layers[0].biases.v[0]);
    clear_model(model);
    save_training_result(result, "loss.csv");
    free(result);
}