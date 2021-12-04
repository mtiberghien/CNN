#include <stdio.h>
#include <stdlib.h>
#include "include/model.h"
#include "math.h"
#include "include/common.h"
#include "include/mnistdata.h"
#include "include/utils.h"
#include "time.h"
#include "omp.h"


void draw_image(tensor* img)
{
    for(int j=0;j<28;j++)
    {
        short is_empty=1;
        for(int i=0;i<28;i++)
        {
            if (img->v[j*28+i]>0)
            {
                is_empty=0;
                break;
            }
        }
        if(!is_empty)
        {
            for(int k=0;k<28;k++)
            {
                double value = img->v[j*28+k];
                printf("\033[%dm  ", value ==0?40:47);
            }
            printf("\033[0m\n");
        }
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
            draw_image(&test->features[i]);
        }
        

    }
    clear_tensors(predictions, test->n_entries);
    free(predictions);
    clear_dataset(test);
    clear_model(model);
}

int main(){
    omp_set_num_threads(10);
    char* filename = "save/model.txt";
    dataset* train = getMNISTData(60000, 0);
    dataset* test = getMNISTData(10000, 1);
    model* model = build_model();
    model->add_layer(build_layer_FC(128, build_activation(RELU)), model);
    model->add_layer(build_layer_FC(10, build_activation(SOFTMAX)), model);
    model->compile(train->features_shape, build_optimizer(ADAM), build_loss(CCE), model);
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