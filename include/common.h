#ifndef COMMON_CNN
#define COMMON_CNN

typedef enum dimension{OneD=1,TwoD=2,ThreeD=3} dimension;
int min(int x,int y);

int index_of(char** list, int n_items, char* item);
void swap(char**, char**);
void sort(char** list, int n_items);

#endif