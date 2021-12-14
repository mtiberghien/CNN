#ifndef UTILS_CNN
#define UTILS_CNN

#include "tensor.h"
#include "dataset.h"
#include "model.h"

extern double evaluate_accuracy(tensor* truth, tensor* prediction, int n_predictions);
extern double evaluate_dataset_accuracy(dataset* data, model* model);
int get_background_color(double gray_scale);
extern void draw_image(tensor* img);
void seconds_to_string(char* string, long seconds);
#endif