#include <CL/sycl.hpp>
#include "CachingAllocator.h"
#include "AllocatorConfig.h"

#include <vector>

int main() {
    sycl::queue queue = sycl::queue{sycl::cpu_selector()};
    unsigned int N = 100;
    std::vector<int> v(N);

    cms::sycltools::CachingAllocator cachingAllocator(queue.get_device(),
        cms::sycltools::config::binGrowth,
        cms::sycltools::config::minBin,
        cms::sycltools::config::maxBin,
        cms::sycltools::config::maxCachedBytes,
        cms::sycltools::config::maxCachedFraction,
        true,
        false);

    void* ptr = cachingAllocator.allocate(N*sizeof(int), queue);
    int* int_ptr = reinterpret_cast<int*>(ptr);

    for (int i=0; i < N; i++) {
      int_ptr[i] = i;
    }

    queue.memcpy(v.data(), ptr, N * sizeof(int));
    queue.wait();

    cachingAllocator.free(ptr);

    for(auto el:v){
      std::cout << el << ", ";
    }
    std::cout << std::endl;

}