#include <stdio.h>
#include <stdlib.h>
#include "include/model.h"
#include "math.h"
#include "include/common.h"
#include "include/mnistdata.h"
#include "include/utils.h"
#include "time.h"

int main(){
    dataset* train = getMNISTData(10000, 0);
    dataset* test = getMNISTData(100, 1);
    model* model = build_model();
    model->add_layer(build_layer_FC(64, build_activation(TANH)), model);
    model->add_layer(build_layer_FC(10, build_activation(TANH)), model);
    model->compile(train->n_features, build_optimizer(ADAM), build_loss(MSE), model);
    training_result* result = model->fit(train->features, train->labels_categorical, train->n_entries, 1, 2, model);
    printf("accuracy training:%6.2f%%\n", evaluate_dataset_accuracy(train, model));
    printf("accuracy test:%6.2f%%\n", evaluate_dataset_accuracy(test, model));
    clear_model(model);
    free(result);
    clear_dataset(train);
    clear_dataset(test);
    
}