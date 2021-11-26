#include "include/utils.h"

double evaluate_accuracy(tensor* truth, tensor* prediction, int n_predictions)
{
    double score=0;
    for(int i=0;i<n_predictions;i++)
    {
        if(arg_max(&truth[i]) == arg_max(&prediction[i]))
        {
            score++;
        }
    }
    return 100*score/n_predictions;
}

double evaluate_dataset_accuracy(dataset* data, model* model)
{
    tensor* predictions = model->predict(data->features, data->n_entries, model);
    return evaluate_accuracy(data->labels_categorical, predictions, data->n_entries);
    clear_tensors(predictions, data->n_entries);
}