#include <AMReX.H>
#include <AMReX_Partition.H>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <thrust/device_malloc_allocator.h>

using namespace amrex;

static constexpr int N = 100'000'000;

void test_amrex (Gpu::DeviceVector<Long>& v)
{
    amrex::Partition(v, [=] AMREX_GPU_DEVICE (const Long& i) noexcept { return i%2 == 0; });
}

struct is_even {
    __device__ bool operator() (const Long& i) { return i%2 == 0; }
};

namespace amrex
{
    template<class T>
    class ThrustAllocator : public thrust::device_malloc_allocator<T>
    {
    public:
        using value_type = T;
        
        typedef thrust::device_ptr<T>  pointer;
        inline pointer allocate(size_t n)
        {
            value_type* result = nullptr;
            result = (value_type*) The_Arena()->alloc(n * sizeof(T));
            return thrust::device_pointer_cast(result);
        }
        
        inline void deallocate(pointer ptr, size_t)
        {
            The_Arena()->free(thrust::raw_pointer_cast(ptr));
        }
    };

    namespace
    {
        ThrustAllocator<char> g_cached_allocator;
    }

    namespace Gpu
    {
        ThrustAllocator<char>& The_ThrustCachedAllocator () { return g_cached_allocator; };
        
        AMREX_FORCE_INLINE auto The_ThrustCachedPolicy() -> decltype (thrust::cuda::par(Gpu::The_ThrustCachedAllocator()))
        {
            return thrust::cuda::par(Gpu::The_ThrustCachedAllocator());
        };
    }
}

void test_vendor (Gpu::DeviceVector<Long>& v)
{
    thrust::partition(Gpu::The_ThrustCachedPolicy(),
                      v.begin(), v.end(), is_even());
    // This is much slower. thrust::partition(thrust::device, v.begin(), v.end(), is_even());
}

void fill_data (Gpu::DeviceVector<Long>& v)
{
    Long* pv = v.data();
    amrex::ParallelFor(v.size(), [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        pv[i] = i*i;
    });
    Gpu::synchronize();
}

int main(int argc, char* argv[])
{
    amrex::Real t_amrex, t_vendor;
    amrex::Initialize(argc,argv);
    {
        Gpu::DeviceVector<Long> v1(N);
        fill_data(v1);

        Gpu::DeviceVector<Long> v2(N);
        fill_data(v2);

        test_amrex(v1);
        fill_data(v1);

        amrex::Real t0 = amrex::second();
        test_amrex(v1);
        t_amrex = amrex::second()-t0;

        test_vendor(v2);
        fill_data(v2);

        t0 = amrex::second();
        test_vendor(v2);
        t_vendor = amrex::second()-t0;

        // DeviceVector uses managed memory
        for (int i = 0; i < N; ++i) {
            if (i < N/2) { // even
                if (v1[i]%2 != 0 || v2[i] % 2 != 0) {
                    std::cout << "WRONG! v1[" << i << "] = " << v1[i] << ", v2[" << i << "] = "
                              << v2[2] << std::endl;
                    break;
                }
            } else {
                if (v1[i]%2 == 0 || v2[i] % 2 == 0) {
                    std::cout << "WRONG! v1[" << i << "] = " << v1[i] << ", v2[" << i << "] = "
                              << v2[2] << std::endl;
                    break;
                }
            }
        }
    }
    amrex::Finalize();
    std::cout << "Kernel run time is " << std::scientific << t_amrex << " and " << t_vendor
              << ".\n";
}
