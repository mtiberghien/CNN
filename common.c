#include <string.h>

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

void swap(char** xp, char** yp)
{
    char* temp = *xp;
    *xp = *yp;
    *yp = temp;
}
 
// A function to implement bubble sort
void sort(char** list, int n_items)
{
   int i, j;
   for (i = 0; i < n_items-1; i++)     
 
       // Last i elements are already in place  
       for (j = 0; j < n_items-i-1; j++)
           if (strcmp(list[j] , list[j+1])>0)
              swap(&list[j], &list[j+1]);
}