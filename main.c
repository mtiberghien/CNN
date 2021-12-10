#include <stdio.h>
#include <stdlib.h>
#include "include/model.h"
#include "math.h"
#include "include/common.h"
#include "include/mnistdata.h"
#include "include/utils.h"
#include "time.h"
#include "omp.h"

//Test predictions on an existing model. The mis predicted images are displayed
void test_model(char* filename, int n_tests)
{
    int test_size=n_tests;
    dataset* test = getMNISTData(test_size, 1);
    model* model = read_model(filename);
    tensor* predictions = model->predict(test->features, test->n_entries, model);
    for(int i=0;i<test->n_entries;i++)
    {
        int label = (int)strtod(test->labels[i],NULL);
        int pred = arg_max(&predictions[i])[0];
        double confidence = 100*predictions[i].v[pred];
        printf("true label:%d, predicted: %d, confidence: %.2lf%%\n", label, pred, confidence);
        if(label!=pred)
        {
            printf("full prediction: ");
            print_tensor(&predictions[i]);
            printf("input image:\n");
            draw_image(&test->features[i]);
        }
    }
    free_tensors(predictions, test->n_entries);
    free_dataset(test);
    free_model(model);
}

void evaluate_model(model* model, int n_entries)
{
    dataset* test = getMNISTData(n_entries,1);
    printf("accuracy test (%d entries):%6.2f%%\n", n_entries, evaluate_dataset_accuracy(test, model));
    free_dataset(test);
}

//Train an existing model providing the filename, the number of entries to use, the batch size and the epochs
void retrain_model(char* filename, int n_entries, int batch_size, int epochs)
{
    dataset* train = getMNISTData(n_entries, 0);
    model* model = read_model(filename);
    training_result* result = model->fit(train->features, train->labels_categorical, train->n_entries, batch_size, epochs, model);
    free_result(result);
    save_model(model, filename);
    evaluate_model(model, 1000);
    free_model(model);
    free_dataset(train);
}

//Display the summary of the model read from the provided filename
void show_model(char* filename)
{
    model* model = read_model(filename);
    model->summary(model);
    free_model(model);
}

//Train MLP architecture using MNIST Data
void train_mlp()
{
    char* filename = "save/mlp_model.txt";
    dataset* train = getMNISTData(60000, 0);
    model* model = build_model();
    model->add_layer(build_layer_Flatten(), model);
    model->add_layer(build_layer_FC(512, build_activation(RELU)), model);
    model->add_layer(build_layer_FC(10, build_activation(SOFTMAX)), model);
    model->compile(train->features_shape, build_optimizer(GD), build_loss(CCE), model);
    model->summary(model);
    save_model(model, filename);
    training_result* result = model->fit(train->features, train->labels_categorical, train->n_entries, 128, 5, model);
    free_result(result);
    save_model(model, filename);
    
    printf("accuracy training:%6.2f%%\n", evaluate_dataset_accuracy(train, model)); 
    evaluate_model(model, 10000);
    
    free_model(model);
    free_dataset(train);
    test_model(filename, 10); 
}

void test_convolutions()
{
    dataset* test = getMNISTData(1,1);
    model* model= build_model();
    model->add_layer(build_layer_Conv2D(5, 3,3, 1, 0, build_activation(RELU)), model);
    model->add_layer(build_layer_MaxPooling2D(2,2,2), model);
    model->compile(test->features_shape, NULL, NULL, model);
    draw_image(test->features);
    tensor* outputs = test->features;
    for(int i=0;i<model->n_layers;i++)
    {
        outputs = model->layers[i].predict(outputs, test->n_entries, &model->layers[i]);
        draw_image(outputs);
        if(i>0)
        {
            model->layers[i-1].clear_predict_memory(&model->layers[i-1]);
        }
    }
    free_tensors(outputs,test->n_entries);
    free_dataset(test);
    free_model(model);
}

//Train a CNN architecture using MNIST Data
void train_cnn()
{
    char* filename = "save/cnn_model_gd.txt";
    dataset* train = getMNISTData(60000, 0);
    model* model = build_model();
    model->add_layer(build_layer_Conv2D(32, 3,3, 1, 0, build_activation(RELU)), model);
    model->add_layer(build_layer_MaxPooling2D(2,2,2), model);
    model->add_layer(build_layer_Conv2D(64, 3,3, 1, 0, build_activation(RELU)), model);
    model->add_layer(build_layer_MaxPooling2D(2,2,2), model);
    model->add_layer(build_layer_Conv2D(64, 3,3, 1, 0, build_activation(RELU)), model);
    model->add_layer(build_layer_Flatten(), model);
    model->add_layer(build_layer_FC(64, build_activation(RELU)), model);
    model->add_layer(build_layer_FC(10, build_activation(SOFTMAX)), model);
    model->compile(train->features_shape, build_optimizer_GD(1E-2,1E-1), build_loss(CCE), model);
    save_model(model, filename);
    show_model(filename);
    training_result* result = model->fit(train->features, train->labels_categorical, train->n_entries, 128, 10, model);
    free_result(result);
    save_model(model, filename);
    
    printf("accuracy training:%6.2f%%\n", evaluate_dataset_accuracy(train, model)); 
    evaluate_model(model, 10000);
    
    free_model(model);
    free_dataset(train);
    test_model(filename, 10); 
}

//Main method for testings
int main(){
    omp_set_num_threads(10);
    model * model = read_model("save/cnn_model_gd.txt");
    evaluate_model(model, 10000);
    free_model(model);
}