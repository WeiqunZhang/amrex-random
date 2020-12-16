#ifndef FOO_H_
#define FOO_H_

#include <AMReX_Gpu.H>

#ifdef AMREX_USE_DPCPP
inline __attribute__((opencl_constant)) int d_a = -1;
inline __attribute__((opencl_constant)) int d_b[] = {-1,-1,-1,-1};
#else
inline AMREX_GPU_CONSTANT int d_a;
inline AMREX_GPU_CONSTANT int d_b[4];
#endif

#endif
