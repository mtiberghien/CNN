#include "include/tensor.h"
#include <stdlib.h>
#include "float.h"
#include <stdio.h>
#include <string.h>
#include "include/common.h"

//Free the memory of a shape
void clear_shape(shape* shape)
{
    free(shape->sizes);
}
//Free the memory of a shape. The shape is destroyed
void free_shape(shape* shape)
{
    clear_shape(shape);
    free(shape);
}
//Free the memory of a 1D tensor
void clear_tensor_1D(double* v, int* sizes)
{
    free(v);
}
//Free the memory of a 2D tensor
void clear_tensor_2D(double* v, int* sizes)
{
    double** matrix = (double**)v;
    int* next = sizes+1;
    for(int i=0;i<*sizes;i++)
    {
        clear_tensor_1D(matrix[i],next);
    }
}
//Free the memory of a 3D tensor
void clear_tensor_3D(double*v, int* sizes)
{
    double*** cube = (double***)v;
    int* next = sizes+1;
    for(int i=0;i<*sizes;i++)
    {
        clear_tensor_2D((double*)cube[i], next);
    }
}

//Free the memory of a tensor
void clear_tensor(tensor* tensor){
    tensor->clear_tensor(tensor->v, tensor->shape->sizes);
    free_shape(tensor->shape);
}
//Free the memory of tensor. The tensor is destroyed
void free_tensor(tensor* tensor)
{
    clear_tensor(tensor);
    free(tensor);
}

//Free memory of a list of tensors
void clear_tensors(tensor* tensors, int n_tensor){
    #pragma omp parallel for
    for(int i=0;i<n_tensor;i++){
        clear_tensor(&tensors[i]);
    }
}
//clear the memory of a list of tensors. The list is also destroyed
void free_tensors(tensor* tensors, int n_tensors)
{
    clear_tensors(tensors, n_tensors);
    free(tensors);
}
//Set each value of a tensor with the result of a function applied to the current value of the tensor
void apply_func(tensor* tensor, double(*func)(double))
{
    int* iterator = get_iterator(tensor);
    int i=0;
    while(!tensor->is_done(tensor, iterator))
    {
        tensor->set_value(tensor, iterator, func(tensor->get_value(tensor, iterator)));
        iterator = tensor->get_next(tensor, iterator);
    }
    free(iterator);
}
//Clone a shape element
shape* clone_shape(const shape* shape)
{
    struct shape* clone = build_shape(shape->dimension);
    for(int i=0;i<clone->dimension;i++)
    {
        clone->sizes[i]=shape->sizes[i];
    }
    return clone;
}
//Multiply each value of a destination tensor by the result of a function applied to the corresponding element of a source tensor
void mult_tensor_func(tensor* tensor_dest,const tensor* tensor_source, double(*func)(double))
{
    int* iterator = get_iterator(tensor_dest);
    while(!tensor_dest->is_done(tensor_dest, iterator))
    {
        tensor_dest->set_value(tensor_dest, iterator, tensor_dest->get_value(tensor_dest, iterator)*func(tensor_source->get_value(tensor_source, iterator)));
        iterator = tensor_dest->get_next(tensor_dest, iterator);
    }
    free(iterator);
}

//Sum the result of a function applied to each element of a tensor
double sum(tensor* tensor, double(*func)(double x))
{
    double result = 0;
    int* iterator = get_iterator(tensor);
    while(!tensor->is_done(tensor, iterator))
    {
        result+=func(tensor->get_value(tensor, iterator));
        iterator = tensor->get_next(tensor, iterator);
    }
    free(iterator);
    return result;
}
//Get the max value of a tensor
double max(tensor* tensor)
{
    double result = -DBL_MAX;
    int* iterator = get_iterator(tensor);
    while(!tensor->is_done(tensor, iterator))
    {
        double v = tensor->get_value(tensor, iterator);
        result = v>result?v:result;
        iterator = tensor->get_next(tensor, iterator);
    }
    free(iterator);
    return result;
}
//Substract the provided value to each element of a tensor
tensor* sub(tensor* tensor, double value)
{
    int* iterator = get_iterator(tensor);
    while(!tensor->is_done(tensor, iterator))
    {
        tensor->set_value(tensor, iterator, tensor->get_value(tensor, iterator)-value);
        iterator = tensor->get_next(tensor, iterator);
    }
    free(iterator);
    return tensor;
}
//Initialize a 1D memory containing double values
double* initialize_array(int size)
{
    return calloc(size, sizeof(double));
}
//Initialize a 2D memory containing double values
double** initialize_matrix(int height, int width)
{
    double ** matrix =(double**)malloc(sizeof(double*)*height); 
    for(int i=0;i<height;i++)
    {
        matrix[i]=initialize_array(width);
    }
    return matrix;
}
//Initialize a 3D memory containing double values
double*** initialize_cube(int height, int width, int depth)
{
    double*** cube = (double***)malloc(sizeof(double**)*depth);
    for(int i=0;i<depth;i++)
    {
        cube[i]=initialize_matrix(height, width);
    }
    return cube;
}
//Get an iterator for the provided tensor
int* get_iterator(const tensor* tensor)
{
    return calloc(tensor->shape->dimension, sizeof(int));
}
//Get the value of a 1D tensor related to the provided iterator
double get_value_1D(const tensor* tensor, int* iterator)
{
    return tensor->v[*iterator];
}
//Set the provided value to a 1D tensor according to the provided iterator
void set_value_1D(const tensor* tensor, int* iterator, double value)
{
    tensor->v[*iterator]=value;
}
//Get the next iterator index for a 1D tensor
int* get_next_1D(const tensor* tensor, int* iterator)
{
    iterator[0]++;
    return iterator;
}
//Get the value of a 2D tensor related to the provided iterator
double get_value_2D(const tensor* tensor, int* iterator)
{
    double** matrix=(double**)tensor->v;
    return matrix[iterator[0]][iterator[1]];
}
//Set the the provided value to a 2D tensor according to the provided iterator
void set_value_2D(const tensor* tensor, int* iterator, double value)
{
    double** matrix=(double**)tensor->v;
    matrix[iterator[0]][iterator[1]] = value;
}
//Get the next iterator indexes for a 2D tensor
int* get_next_2D(const tensor* tensor, int* iterator)
{
    iterator[1]++;
    if(iterator[1] >=  tensor->shape->sizes[1])
    {
        iterator[0]++;
        iterator[1]=0;
    }
    return iterator;
}
//Get the value of tensor related to the provided iterator
double get_value_3D(const tensor* tensor, int* iterator)
{
    double*** cube=(double***)tensor->v;
    return cube[iterator[0]][iterator[1]][iterator[2]];
}
//Set the provided value into the tensor according to the provided iterator
void set_value_3D(const tensor* tensor, int* iterator, double value)
{
    double*** cube=(double***)tensor->v;
    cube[iterator[0]][iterator[1]][iterator[2]] = value;
}
//Get the next iterator indexes for a 3D tensor
int* get_next_3D(const tensor* tensor, int* iterator)
{
    iterator[2]++;
    if(iterator[2] >=  tensor->shape->sizes[2])
    {
        iterator[1]++;
        iterator[2]=0;
        if(iterator[1] >= tensor->shape->sizes[1])
        {
            iterator[0]++;
            iterator[1]=0;
        }
    }
    return iterator;
}
//Return 1 if an iterator has reach the end of a tensor 0 otherwise
short is_done(const tensor* tensor, int* iterator)
{
    return tensor->shape->sizes[0] <= iterator[0];
}
//Intialize the memory and methods for a 1D tensor
void initialize_tensor_1D(tensor* tensor)
{
    tensor->v=initialize_array(*tensor->shape->sizes);
    tensor->get_value = get_value_1D;
    tensor->set_value = set_value_1D;
    tensor->get_next = get_next_1D;
    tensor->is_done = is_done;
    tensor->clear_tensor = clear_tensor_1D;
}
//Initialize the memory and methods for a 2D tensor
void initialize_tensor_2D(tensor* tensor)
{
    int height = tensor->shape->sizes[0];
    int width = tensor->shape->sizes[1];
    
    tensor->v=(double*)initialize_matrix(height,width);
    tensor->get_value = get_value_2D;
    tensor->set_value = set_value_2D;
    tensor->get_next = get_next_2D;
    tensor->is_done = is_done;
    tensor->clear_tensor = clear_tensor_2D;
}
//Intialize the memory and methods for a 3D tensor
void initialize_tensor_3D(tensor* tensor)
{
    int depth = tensor->shape->sizes[0];
    int height = tensor->shape->sizes[1];
    int width = tensor->shape->sizes[2];
    tensor->v=(double*)initialize_cube(height,width, depth);
    tensor->get_value = get_value_3D;
    tensor->set_value = set_value_3D;
    tensor->get_next = get_next_3D;
    tensor->is_done = is_done;
    tensor->clear_tensor = clear_tensor_3D;
    
}
//Initialize the memory of a shape
shape* build_shape(dimension dim)
{
    shape* result = (shape*)malloc(sizeof(shape));
    result->dimension = dim;
    result->sizes = (int*)calloc(dim, sizeof(int));
    return result;
}
//Intialize the memory of a tensor according to the provided shape
void initialize_tensor(tensor* tensor, shape* shape)
{
    tensor->shape = clone_shape(shape);
    switch(shape->dimension)
    {
        case TwoD: initialize_tensor_2D(tensor);break;
        case ThreeD: initialize_tensor_3D(tensor);break;
        default: initialize_tensor_1D(tensor);break;
    }
}
//Print a 1D tensor to the standard output
void print_tensor_1D(double* values, int* size)
{
    int length =*size;
    printf("[");
    for(int i=0;i<length;i++)
    {
        printf("%lf", values[i]);
        if(i!=length-1)
        {
            printf(",");
        }
    }
    printf("]");
}
//Print a 2D tensor to the standard output
void print_tensor_2D(double* values, int* size)
{
    int height = size[0];
    int width = size[1];
    double** matrix = (double**)values;
    printf("[");
    for(int i=0;i<height;i++)
    {
        print_tensor_1D(matrix[i], &width);
        if(i!=height-1)
        {
            printf(",\n");
        }
    }
    printf("]");
}
//Print a 3D tensor to the standard output
void print_tensor_3D(double* values, int * size)
{
    int depth= size[0];
    int height= size[1];
    int width = size[2];
    double*** cube= (double***)values;
    int s2D[2]={height,width};
    printf("[");
    for(int i=0;i<depth;i++)
    {
        print_tensor_2D((double*)cube[i],s2D);
        if(i!=depth-1)
        {
            printf(",\n");
        }
    }
    printf("]");
}
// Print a tensor to the standard output
void print_tensor(const tensor* tensor)
{
    switch (tensor->shape->dimension)
    {
        case TwoD: print_tensor_2D(tensor->v, tensor->shape->sizes); break;
        case ThreeD: print_tensor_3D(tensor->v, tensor->shape->sizes); break;
        default: print_tensor_1D(tensor->v, tensor->shape->sizes);break;
    }
    printf("\n");
}
//Transform an array of labels to a categorical tensor
tensor* to_categorical(char** labels, int n_labels)
{
    char** unique_labels = (char**)malloc(sizeof(char*));
    int count=0;
    for(int i=0;i<n_labels;i++)
    {
        if(index_of(unique_labels, count, labels[i])<0)
        {
            unique_labels = realloc(unique_labels, sizeof(char*)*++count);
            unique_labels[count-1]=labels[i];
        }
    }
    sort(unique_labels, count);
    tensor* result = malloc(sizeof(tensor)*n_labels);
    shape* shape = build_shape(OneD);
    shape->sizes[0]=count;
    for(int i=0;i<n_labels;i++)
    {
        initialize_tensor(&result[i], shape);
        result[i].v[index_of(unique_labels, count, labels[i])]=1;
    }
    free(unique_labels);
    return result;
}
//Get the iterator for the maximum value of a tensor
int* arg_max(tensor* tensor)
{
    double max = -DBL_MAX;
    int* result=malloc(sizeof(int)*tensor->shape->dimension);
    for(int i=0;i<tensor->shape->dimension;i++)
    {
        result[i]=-1;
    }
    int* iterator = get_iterator(tensor);

    while(!tensor->is_done(tensor, iterator))
    {
        double v = tensor->get_value(tensor, iterator);
        if(v>max)
        {
            max = v;
            for(int i=0;i<tensor->shape->dimension;i++)
            {
                result[i]=iterator[i];
            }
        }
        iterator = tensor->get_next(tensor, iterator);
    }
    free(iterator);
    return result;
}

//Check if an iterator over a tensor is equal to another iterator for each dimension
short is_iterator_equal(int* i1, int*i2, dimension dimension)
{
    for(int i=0;i<dimension;i++)
    {
        if(i1[0]!=i2[0])
        {
            return 0;
        }
    }
    return 1;
}
//Write a shape to a file
void save_shape(FILE* fp, shape* shape)
{
    int dim = shape->dimension;
    int* sizes = shape->sizes;
    fprintf(fp, "{dim:%d, sizes:", dim);
    for(int i=0;i<dim;i++)
    {
        fprintf(fp, "%d", sizes[i]);
        if(i!=dim-1)
        {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "}");
}
//Read a shape from file
shape* read_shape(FILE* fp)
{
    int dimension;
    fscanf(fp, "{dim:%d, sizes:", &dimension);
    shape* result = build_shape(dimension);
    for(int i=0;i<dimension;i++)
    {
        fscanf(fp, "%d", &result->sizes[i]);
        if(i!=dimension-1)
        {
            fscanf(fp, ",");
        }
    }
    fscanf(fp, "}");
    return result;
}

//Write a tensor to file
void save_tensor(FILE* fp, tensor* tensor)
{
    int* iterator = get_iterator(tensor);
    short is_done = tensor->is_done(tensor, iterator);
    while(!is_done)
    {
        fprintf(fp, "%le", tensor->get_value(tensor, iterator));
        iterator = tensor->get_next(tensor, iterator);
        is_done = tensor->is_done(tensor, iterator);
        if(!is_done)
        {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "\n");
    free(iterator);    
}

//Read a tensor from file
void read_tensor(FILE* fp, tensor* tensor)
{
    int* iterator = get_iterator(tensor);
    short is_done = tensor->is_done(tensor, iterator);
    while(!is_done)
    {
        double d;
        fscanf(fp, "%le",&d);
        tensor->set_value(tensor, iterator, d);
        iterator = tensor->get_next(tensor, iterator);
        is_done = tensor->is_done(tensor, iterator);
        if(!is_done)
        {
            fscanf(fp, ",");
        }
    }
    fscanf(fp, "\n");
    free(iterator);
}


