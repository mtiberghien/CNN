#include <stdio.h>
#include <stdlib.h>
#include "include/model.h"
#include "math.h"
#include "include/common.h"
#include "include/mnistdata.h"
#include "include/utils.h"
#include "time.h"
#include "omp.h"

int main(){
    char* filename = "save/model2.txt";
    dataset* train = getMNISTData(60000, 0);
    dataset* test = getMNISTData(10000, 1);
    model* model = build_model();
    model->add_layer(build_layer(FC, 128, build_activation(RELU)), model);
    model->add_layer(build_layer(FC, 10, build_activation(SOFTMAX)), model);
    model->compile(train->n_features, build_optimizer(ADAM), build_loss(CCE), model);
    save_model(model, filename);
    training_result* result = model->fit(train->features, train->labels_categorical, train->n_entries, 128, 5, model);
    free(result);
    save_model(model, filename);
    
    printf("accuracy training:%6.2f%%\n", evaluate_dataset_accuracy(train, model));
    printf("accuracy test:%6.2f%%\n", evaluate_dataset_accuracy(test, model));
    
    
    clear_dataset(test);
    clear_model(model);
    clear_dataset(train);
}