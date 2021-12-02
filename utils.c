#include "include/utils.h"
#include "include/tensor.h"

double evaluate_accuracy(tensor* truth, tensor* prediction, int n_predictions)
{
    double score=0;
    for(int i=0;i<n_predictions;i++)
    {
        int* it_truth = arg_max(&truth[i]);
        int* it_prediction = arg_max(&prediction[i]);
        if(is_iterator_equal(it_truth,it_prediction, truth[i].shape->dimension))
        {
            score++;
        }
        free(it_truth);
        free(it_prediction);
    }
    return 100*score/n_predictions;
}

double evaluate_dataset_accuracy(dataset* data, model* model)
{
    tensor* predictions = model->predict(data->features, data->n_entries, model);
    double result = evaluate_accuracy(data->labels_categorical, predictions, data->n_entries);
    clear_model_predict_memory(model);
    return result;
}