#include <AMReX.H>
#include <AMReX_Scan.H>

using namespace amrex;

static constexpr int N = 100'000'000;

void test_amrex (Gpu::DeviceVector<Long>& result)
{
    Long* pr = result.data();
    Scan::PrefixSum<int>(N,
                         [=] AMREX_GPU_DEVICE (int i) noexcept
                         {
                             return i%2 == 0;
                         },
                         [=] AMREX_GPU_DEVICE (int i, int psum) noexcept
                         {
                             if (i%2 == 0) {
                                 auto L = static_cast<Long>(i);
                                 pr[psum] = L*L;
                             }
                         },
                         Scan::Type::exclusive, Scan::noRetSum);
}

void test_vendor (Gpu::DeviceVector<Long>& result)
{
    Long* pr = result.data();
    Gpu::DeviceVector<int> in(N);
    Gpu::DeviceVector<int> out(N);
    int* pin = in.data();
    amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        pin[i] = i;
    });
    Gpu::exclusive_scan(in.begin(), in.end(), out.begin());
    int* pout = out.data();
    amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        if (i%2 == 0) {
            auto L = static_cast<Long>(i);
            pr[pout[i]] = L*L;
        }        
    });
    Gpu::synchronize();
}

int main(int argc, char* argv[])
{
    amrex::Real t_amrex, t_vendor;
    amrex::Initialize(argc,argv);
    {
        Gpu::DeviceVector<Long> result(N);

        test_amrex(result);

        amrex::Real t0 = amrex::second();
        test_amrex(result);
        t_amrex = amrex::second()-t0;

        test_vendor(result);
        t0 = amrex::second();
        test_vendor(result);
        t_vendor = amrex::second()-t0;
    }
    amrex::Finalize();
    std::cout << "Kernel run time is " << std::scientific << t_amrex << " and " << t_vendor
              << ".\n";
}
