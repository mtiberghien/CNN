#include <stdio.h>
#include <stdlib.h>
#include "include/model.h"
#include "math.h"
#include "include/common.h"
#include "include/mnistdata.h"
#include "include/utils.h"
#include "time.h"

int main(){
    dataset* train = getMNISTData(60000, 0);
    dataset* test = getMNISTData(10000, 1);
    model* model = read_model("save/model.txt");
    model->predict(train->features, train->n_entries, model);
    clear_model(model);
    /*
    model* model = build_model();
    model->add_layer(build_layer(FC, 512, build_activation(TANH)), model);
    model->add_layer(build_layer(FC, 10, build_activation(TANH)), model);
    model->compile(train->n_features, build_optimizer(GD), build_loss(MSE), model);
    save_model(model, "save/model2.txt");
    training_result* result = model->fit(train->features, train->labels_categorical, train->n_entries, 128, 1, model);
    free(result);
    printf("accuracy training:%6.2f%%\n", evaluate_dataset_accuracy(train, model));
    printf("accuracy test:%6.2f%%\n", evaluate_dataset_accuracy(test, model));
    save_model(model, "save/model2.txt");
    clear_model(model);
    clear_dataset(train);
    clear_dataset(test);
    */
}