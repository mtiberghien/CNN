#ifndef PROGRESSION_CNN
#define PROGRESSION_CNN

typedef struct progression{
    int step;
    int total_steps;
    char* header;
    void (*call_back)(struct progression*);
    void (*done)(struct progression*);
} progression;

progression* build_progression(int total_steps, char* header);
void free_progression(progression*);
#endif