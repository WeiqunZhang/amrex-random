#include <AMReX.H>
#include <AMReX_Scan.H>

using namespace amrex;

static constexpr int N = 100'000'000;

// Conceptually in this test, we scan N ints and remove all the odd numbers and move all
// the even numbers to the front keeping their original order.  Then we square them and
// save the results in a vector.  This is not a real example in AMReX.  But we do similar
// things in AMReX.  This example is to demonstrate that it is desirable to have a
// flexible interface for the scan function to take lambda funcitons instead of iterators.
//
// Note that we could use a fancy iterator that combines transform iterator
// and counting iterator for input instead of a lambda function.  However,
// we cannot do this for the output, because a transgorm iterator only takes
// a unary function, whereas our lambda function for output takes two
// arguments, the index and the scan result.
//
// Another feature that could be useful is an option to return the
// aggregated final result so that we can easily get the total sum
// (especially for exclusive scan).

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
    Gpu::DeviceVector<int> inout(N);
    int* pinout = inout.data();
    amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        pinout[i] = (i%2 == 0);
    });
    Gpu::exclusive_scan(inout.begin(), inout.end(), inout.begin());
    amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        if (i%2 == 0) {
            auto L = static_cast<Long>(i);
            pr[pinout[i]] = L*L;
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
