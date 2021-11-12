#include "include/tensor.h"
#include <stdlib.h>

void clear_tensor(tensor tensor){
    free(tensor.t);
}