#ifndef FOO_H_
#define FOO_H_

#include <AMReX_Gpu.H>

extern AMREX_GPU_CONSTANT int d_a;
extern AMREX_GPU_CONSTANT int d_b[];

AMREX_GPU_DEVICE AMREX_INLINE
void foo (int* p)
{
    p[0] = d_a;
    p[1] = d_b[0];
    p[2] = d_b[1];
    p[3] = d_b[2];
}

#endif
