#include "include/progression.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void progression_callback(progression* progression)
{
    progression->step++;
    double percentage = (double)(progression->step*100)/progression->total_steps;
    printf("\033[K\r%s: %.2f%%", progression->header, percentage);
    fflush(stdout);
}

void progression_done(progression* progression)
{
    printf("\n");
    progression->step=0;
}

void clear_progression(progression* progression)
{
    progression->done(progression);
    free(progression);
}

progression* build_progression(int total_steps, char* header)
{
    progression* progression = (struct progression*)malloc(sizeof(struct progression));
    progression->step=0;
    progression->total_steps=total_steps;
    progression->header=header;
    progression->call_back=progression_callback;
    progression->done=progression_done;
}