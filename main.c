#include <stdio.h>
#include <stdlib.h>
#include "include/model.h"
#include "math.h"
#include "include/common.h"
#include "include/mnistdata.h"
#include "include/utils.h"
#include "time.h"
#include "omp.h"

void test_model(char* filename)
{
    int test_size=10;
    dataset* test = getMNISTData(test_size, 1);
    model* model = read_model(filename);
    tensor* predictions = model->predict(test->features, test->n_entries, model);
    for(int i=0;i<test->n_entries;i++)
    {
        int label = (int)strtod(test->labels[i],NULL);
        int pred = arg_max(&predictions[i])[0];
        printf("true label:%d, predicted: %d\n", label, pred);
        if(label!=pred)
        {
            printf("full prediction: ");
            print_tensor(&predictions[i]);
            printf("input image:\n");
            draw_image(&test->features[i]);
        }
        

    }
    clear_tensors(predictions, test->n_entries);
    free(predictions);
    clear_dataset(test);
    clear_model(model);
}

void show_model(char* filename)
{
    model* model = read_model(filename);
    model->summary(model);
    clear_model(model);
}

void train_mlp()
{
    char* filename = "save/mlp_model.txt";
    dataset* train = getMNISTData(10000, 0);
    dataset* test = getMNISTData(10000, 1);
    model* model = build_model();
    model->add_layer(build_layer_Flatten(), model);
    model->add_layer(build_layer_FC(64, build_activation(RELU)), model);
    model->add_layer(build_layer_FC(10, build_activation(SOFTMAX)), model);
    model->compile(train->features_shape, build_optimizer(ADAM), build_loss(CCE), model);
    model->summary(model);
    save_model(model, filename);
    training_result* result = model->fit(train->features, train->labels_categorical, train->n_entries, 64, 5, model);
    clear_result(result);
    free(result);
    save_model(model, filename);
    
    printf("accuracy training:%6.2f%%\n", evaluate_dataset_accuracy(train, model)); 
    printf("accuracy test:%6.2f%%\n", evaluate_dataset_accuracy(test, model));
    
    clear_model(model);
    clear_dataset(train);
    clear_dataset(test);
    test_model(filename); 
}

void test()
{
    char* filename = "save/cnn_model.txt";
    dataset* test = getMNISTData(100, 1);
    model* model = read_model(filename);
    printf("Label:%s\n", test->labels[0]);
    draw_image(test->features);
    show_model(filename);
    training_result* result = model->fit(test->features, test->labels_categorical, 1,1, 1, model);
    clear_result(result);
    free(result);   
    clear_model(model);
    clear_dataset(test);
}

void retrain_model(char* filename, int n_entries, int batch_size, int epochs)
{
    dataset* train = getMNISTData(n_entries, 0);
    dataset* test = getMNISTData(1000, 1);
    model* model = read_model(filename);
    training_result* result = model->fit(train->features, train->labels_categorical, train->n_entries, batch_size, epochs, model);
    clear_result(result);
    free(result);
    save_model(model, filename);
    printf("accuracy test:%6.2f%%\n", evaluate_dataset_accuracy(test, model));
    clear_model(model);
    clear_dataset(train);
    clear_dataset(test);
}

void train_cnn()
{
    char* filename = "save/cnn_model.txt";
    dataset* train = getMNISTData(10000, 0);
    dataset* test = getMNISTData(1000, 1);
    model* model = build_model();
    model->add_layer(build_layer_Conv2D(32, 3,3, 1, 0, build_activation(RELU)), model);
    model->add_layer(build_layer_MaxPooling2D(2,2,2), model);
    model->add_layer(build_layer_Conv2D(64, 3,3, 1, 0, build_activation(RELU)), model);
    model->add_layer(build_layer_MaxPooling2D(2,2,2), model);
    model->add_layer(build_layer_Conv2D(64, 3,3, 1, 0, build_activation(RELU)), model);
    model->add_layer(build_layer_Flatten(), model);
    model->add_layer(build_layer_FC(64, build_activation(RELU)), model);
    model->add_layer(build_layer_FC(10, build_activation(SOFTMAX)), model);
    model->compile(train->features_shape, build_optimizer(ADAM), build_loss(CCE), model);
    save_model(model, filename);
    show_model(filename);
    training_result* result = model->fit(train->features, train->labels_categorical, train->n_entries, 128, 10, model);
    clear_result(result);
    free(result);
    save_model(model, filename);
    
    printf("accuracy training:%6.2f%%\n", evaluate_dataset_accuracy(train, model)); 
    printf("accuracy test:%6.2f%%\n", evaluate_dataset_accuracy(test, model));
    
    clear_model(model);
    clear_dataset(train);
    clear_dataset(test);
    test_model(filename); 
}

int main(){
    omp_set_num_threads(10);
    retrain_model("save/mlp_model.txt", 60000, 128,10);
}