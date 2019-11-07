#include <iostream>

#include "mbpol.h"

namespace {

static x2o::mbpol model;

#ifdef _OPENMP
#  pragma omp threadprivate(model)
#endif

} // namespace

extern "C" {

void calcpot_(int* nw, double* Vpot, const double* x)
{
    *Vpot = model(*nw, x);
}

void calcpotg_(int* nw, double* Vpot, const double* x, double* g)
{
    *Vpot = model(*nw, x, g);
}

} // extern "C"
