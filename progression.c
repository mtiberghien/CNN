#include "include/progression.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

//The method is called at each step of a progression
void progression_callback(progression* progression)
{
    #pragma omp critical
    {
        progression->step++;
        double percentage = (double)(progression->step*100)/progression->total_steps;
        printf("\033[K\r%s: %.2f%%", progression->header, percentage);
        fflush(stdout);
    }
}
//The methods is executed when the progression object is destroyed
void progression_done(progression* progression)
{
    printf("\n");
    progression->step=0;
}

//Execute the done function. The progression object is destroyed
void free_progression(progression* progression)
{
    progression->done(progression);
    free(progression);
}

//Initialize the memory and methods of a progression object.
progression* build_progression(int total_steps, char* header)
{
    progression* progression = (struct progression*)malloc(sizeof(struct progression));
    progression->step=0;
    progression->total_steps=total_steps;
    progression->header=header;
    progression->call_back=progression_callback;
    progression->done=progression_done;
}