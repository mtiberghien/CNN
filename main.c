#include <stdio.h>
#include <stdlib.h>
#include "include/tensor.h"

int remove_line(tensor* array, int index, int size_array);
int remove_line(tensor* array, int index, int size_array)
{
    if(index<size_array && index>=0)
    {
        clear_tensor(&array[index]);
        for(int i=index;i<size_array-1;i++)
        {
            array[i]=array[i+1];
        }
        array=realloc(array,(size_array-1)*sizeof(tensor));
        return size_array-1;
    }
    return size_array;
}

int main(){
    tensor* array= (tensor*)malloc(sizeof(tensor));
    int size_array = 3;
    for(int i=0;i<size_array;i++)
    {
        array = realloc(array,(i+1)*sizeof(tensor));
        tensor* t = &array[i];
        t->size = 3;
        t->v=calloc(t->size, sizeof(double));
        for(int j=0;j<t->size;j++)
        {
            t->v[j]=i*t->size+j;
        }
    }

    size_array = remove_line(array,1, size_array);

    for(int i=0;i<size_array;i++)
    {
        for(int j=0;j<3;j++)
        {
            printf("%f", array[i].v[j]);
            if(j<2)
            {
                printf(";");
            }
        }
        printf("\n");
    }

    clear_tensors(array, size_array);
    free(array);


    
    
}