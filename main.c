#include <stdio.h>
#include <stdlib.h>
#include "include/model.h"
#include "math.h"
#include "include/common.h"
#include "include/mnistdata.h"
#include "include/utils.h"
#include "time.h"
#include "omp.h"

int get_background_color(double gray_scale)
{
    return 232 + (int)(gray_scale*24);
}

void draw_mnist_image(tensor* img)
{
    double** img_matrix = ((double***)img->v)[0];
    for(int j=0;j<28;j++)
    {
        for(int k=0;k<28;k++)
        {
            double value = img_matrix[j][k];
            printf("\033[48;5;%dm  ", get_background_color(value));
        }
        printf("\033[0m\n");
    }
}

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
            draw_mnist_image(&test->features[i]);
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
    dataset* train = getMNISTData(60000, 0);
    dataset* test = getMNISTData(10000, 1);
    model* model = build_model();
    model->add_layer(build_layer_Flatten(), model);
    model->add_layer(build_layer_FC(128, build_activation(RELU)), model);
    model->add_layer(build_layer_FC(10, build_activation(SOFTMAX)), model);
    model->compile(train->features_shape, build_optimizer(ADAM), build_loss(CCE), model);
    model->summary(model);
    save_model(model, filename);
    training_result* result = model->fit(train->features, train->labels_categorical, train->n_entries, 128, 5, model);
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

void test_flatten()
{
    shape* shape = build_shape(ThreeD);
    shape->sizes[0]=2;
    shape->sizes[1]=2;
    shape->sizes[2]=3;
    tensor test;
    initialize_tensor(&test, shape);
    clear_shape(shape);
    free(shape);
    int* iterator = get_iterator(&test);
    int i=0;
    while(!test.is_done(&test, iterator))
    {
        test.set_value(&test, iterator, i++);
        iterator = test.get_next(&test, iterator);
    }
    free(iterator);
    model* model = build_model();
    model->add_layer(build_layer_Flatten(), model);
    model->compile(test.shape, build_optimizer(GD), build_loss(MSE), model);
    tensor truth;
    initialize_tensor(&truth, model->layers[0].output_shape);
    model->fit(&test, &truth, 1, 1, 1, model);
    clear_model(model);
}

void test_Conv()
{
    shape* shape = build_shape(ThreeD);
    shape->sizes[0]=2;
    shape->sizes[1]=5;
    shape->sizes[2]=5;
    tensor test;
    initialize_tensor(&test, shape);
    clear_shape(shape);
    free(shape);
    int* iterator=get_iterator(&test);
    int i=0;
    while(!test.is_done(&test, iterator))
    {
        test.set_value(&test, iterator, 10E-3*i++);
        iterator = test.get_next(&test, iterator);
    }
    free(iterator);
    print_tensor(&test);
    model* model = build_model();
    model->add_layer(build_layer_Conv2D(3,3,3,1,0, build_activation(RELU)), model);
    model->compile(test.shape, build_optimizer(GD), build_loss(MSE), model);
    tensor truth;
    initialize_tensor(&truth, model->layers[0].output_shape);
    model->fit(&test, &truth,1,1,1, model);
    clear_model(model);
}

void train_cnn()
{
    char* filename = "save/cnn_model.txt";
    dataset* train = getMNISTData(10000, 0);
    dataset* test = getMNISTData(1000, 1);
    model* model = build_model();
    model->add_layer(build_layer_Conv2D(32, 3,3, 1, 0, build_activation(RELU)), model);
    model->add_layer(build_layer_MaxPooling2D(2,2,2), model);
    model->add_layer(build_layer_Flatten(), model);
    model->add_layer(build_layer_FC(64, build_activation(RELU)), model);
    model->add_layer(build_layer_FC(10, build_activation(SOFTMAX)), model);
    model->compile(train->features_shape, build_optimizer(ADAM), build_loss(CCE), model);
    save_model(model, filename);
    show_model(filename);
    training_result* result = model->fit(train->features, train->labels_categorical, train->n_entries, 32, 1, model);
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
    train_cnn();
}