#include "string.h"

int min(int x,int y)
{
    return  x<y?x:y;
}

int index_of(char** list, int n_items, char* item)
{
    for(int i=0;i<n_items;i++)
    {
        if (strcmp(list[i], item) == 0)
        {
            return i;
        }
    }
    return -1;
}

void sort(char** list, int n_items)
{
    for(int i =0;i<n_items;i++)
    {
        //TODO
    }
}