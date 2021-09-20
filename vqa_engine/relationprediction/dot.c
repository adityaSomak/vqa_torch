#include <stdio.h>
#include "dot.h"

double dot(int len, double* vec1, double* vec2)
{
    int i;
    double d;

    d = 0;
    for(i=0;i<len;i++)
        d += vec1[i]*vec2[i];

    return d;
}