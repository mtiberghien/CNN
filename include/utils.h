#ifndef UTILS_CNN
#define UTILS_CNN

#include "tensor.h"
#include "dataset.h"
#include "model.h"

double evaluate_accuracy(tensor* truth, tensor* prediction, int n_predictions);
double evaluate_dataset_accuracy(dataset* data, model* model);
int get_background_color(double gray_scale);
void draw_image(tensor* img);
#endif