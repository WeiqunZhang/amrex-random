#include <foo.H>

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_Print.H>

using namespace amrex;

#if defined(AMREX_USE_CUDA)
#  define AMREX_GPU_MEMCPY_TO_SYMBOL(d,h,n) cudaMemcpyToSymbol(d, h, n);
#elif defined(AMREX_USE_HIP)
#  define AMREX_GPU_MEMCPY_TO_SYMBOL(d,h,n) hipMemcpyToSymbol(d, h, n);
#elif defined(AMREX_USE_DPCPP)
#  define AMREX_GPU_MEMCPY_TO_SYMBOL(d,h,n) 
#else
#  define AMREX_GPU_MEMCPY_TO_SYMBOL(d,h,n) std::memcpy(&d, h, n);
#endif

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int h_a = 3;
        int h_b[] = {10,20,30,40};

        AMREX_GPU_MEMCPY_TO_SYMBOL(d_a, &h_a, sizeof(int));
        AMREX_GPU_MEMCPY_TO_SYMBOL(d_b, h_b, sizeof(int)*4);

        Gpu::PinnedVector<int> pv(4);
        int* ppv = pv.data();

        amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int)
        {
            foo(ppv);
        });

        Gpu::synchronize();

        amrex::Print() << pv[0] << ", " << pv[1] << ", " << pv[2] << ", " << pv[3] << std::endl;
    }
    amrex::Finalize();
}
