#include "include/tensor.h"
#include <stdlib.h>
#include "float.h"
#include <stdio.h>
#include <string.h>
#include "include/common.h"


void clear_shape(shape* shape)
{
    free(shape->sizes);
}

void clear_tensor_1D(double* v, int* sizes)
{
    free(v);
}

void clear_tensor_2D(double* v, int* sizes)
{
    double** matrix = (double**)v;
    int* next = sizes+1;
    for(int i=0;i<*sizes;i++)
    {
        clear_tensor_1D(matrix[i],next);
    }
}

void clear_tensor_3D(double*v, int* sizes)
{
    double*** cube = (double***)v;
    int* next = sizes+1;
    for(int i=0;i<*sizes;i++)
    {
        clear_tensor_2D((double*)cube[i], next);
    }
}

//Free memory of a tensor
void clear_tensor(tensor* tensor){
    tensor->clear_tensor(tensor->v, tensor->shape->sizes);
    clear_shape(tensor->shape);
    free(tensor->shape);
}

//Free memory for a batch of tensors
void clear_tensors(tensor* tensors, int n_tensor){
    #pragma omp parallel for
    for(int i=0;i<n_tensor;i++){
        clear_tensor(&tensors[i]);
    }
}

void apply_func(tensor* tensor, double(*func)(double))
{
    int* iterator = get_iterator(tensor);
    while(!tensor->is_done(tensor, iterator))
    {
        tensor->set_value(tensor, iterator, func(tensor->get_value(tensor, iterator)));
        iterator = tensor->get_next(tensor, iterator);
    }
    free(iterator);
}

shape* clone_shape(const shape* shape)
{
    struct shape* clone = build_shape(shape->dimension);
    for(int i=0;i<clone->dimension;i++)
    {
        clone->sizes[i]=shape->sizes[i];
    }
    return clone;
}

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

//Sum the result of a function on a tensor
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

double* initialize_array(int size)
{
    return calloc(size, sizeof(double));
}

double** initialize_matrix(int height, int width)
{
    double ** matrix =(double**)malloc(sizeof(double*)*height); 
    for(int i=0;i<height;i++)
    {
        matrix[i]=initialize_array(width);
    }
    return matrix;
}

double*** initialize_cube(int height, int width, int depth)
{
    double*** cube = (double***)malloc(sizeof(double**)*depth);
    for(int i=0;i<depth;i++)
    {
        cube[i]=initialize_matrix(height, width);
    }
    return cube;
}

int* get_iterator(const tensor* tensor)
{
    return calloc(tensor->shape->dimension, sizeof(int));
}

double get_value_1D(const tensor* tensor, int* iterator)
{
    return tensor->v[*iterator];
}

void set_value_1D(const tensor* tensor, int* iterator, double value)
{
    tensor->v[*iterator]=value;
}

int* get_next_1D(const tensor* tensor, int* iterator)
{
    iterator[0]++;
    return iterator;
}

double get_value_2D(const tensor* tensor, int* iterator)
{
    double** matrix=(double**)tensor->v;
    return matrix[iterator[0]][iterator[1]];
}

void set_value_2D(const tensor* tensor, int* iterator, double value)
{
    double** matrix=(double**)tensor->v;
    matrix[iterator[0]][iterator[1]] = value;
}

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

double get_value_3D(const tensor* tensor, int* iterator)
{
    double*** cube=(double***)tensor->v;
    return cube[iterator[0]][iterator[1]][iterator[2]];
}

void set_value_3D(const tensor* tensor, int* iterator, double value)
{
    double*** cube=(double***)tensor->v;
    cube[iterator[0]][iterator[1]][iterator[2]] = value;
}

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

short is_done(const tensor* tensor, int* iterator)
{
    return tensor->shape->sizes[0] <= iterator[0];
}

void initialize_tensor_1D(tensor* tensor)
{
    tensor->v=initialize_array(*tensor->shape->sizes);
    tensor->get_value = get_value_1D;
    tensor->set_value = set_value_1D;
    tensor->get_next = get_next_1D;
    tensor->is_done = is_done;
    tensor->clear_tensor = clear_tensor_1D;
}

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

shape* build_shape(dimension dim)
{
    shape* result = (shape*)malloc(sizeof(shape));
    result->dimension = dim;
    result->sizes = (int*)calloc(dim, sizeof(int));
    return result;
}

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

void save_tensor(FILE* fp, tensor* tensor)
{
    int* iterator = get_iterator(tensor);
    short is_done = tensor->is_done(tensor, iterator);
    while(!is_done)
    {
        fprintf(fp, "%lf", tensor->get_value(tensor, iterator));
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

void read_tensor(FILE* fp, tensor* tensor)
{
    int* iterator = get_iterator(tensor);
    short is_done = tensor->is_done(tensor, iterator);
    while(!is_done)
    {
        double d;
        fscanf(fp, "%lf",&d);
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


