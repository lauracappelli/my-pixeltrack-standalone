#include <CL/sycl.hpp>
#include "SYCLCore/CachingAllocator.h"
#include "SYCLCore/AllocatorConfig.h"

#include <vector>

int main() {
    sycl::queue queue = sycl::queue{sycl::cpu_selector()};
    unsigned int N = 100;
    std::vector<int> v(N);
    std::vector<int> v2(N);

    cms::sycltools::CachingAllocator cachingAllocator(queue.get_device(),
        cms::sycltools::config::binGrowth,
        cms::sycltools::config::minBin,
        cms::sycltools::config::maxBin,
        cms::sycltools::config::maxCachedBytes,
        cms::sycltools::config::maxCachedFraction,
        true,
        true);

    void* ptr = cachingAllocator.allocate(N*sizeof(int), queue);
    int* int_ptr = reinterpret_cast<int*>(ptr);

    for (auto i=0; i < N; i++) {
      int_ptr[i] = i;
    }

    queue.memcpy(v.data(), ptr, N * sizeof(int));
    queue.wait();
    
    // for(auto el:v){
    //   std::cout << el << ", ";
    // }
    // std::cout << std::endl;

    void* ptr2 = cachingAllocator.allocate(N*sizeof(int), queue);
    int* int_ptr2 = reinterpret_cast<int*>(ptr2);

    for (auto i=0; i < N; i++) {
      int_ptr2[i] = 42;
    }

    queue.memcpy(v2.data(), ptr2, N* sizeof(int));
    queue.wait();

    cachingAllocator.free(ptr);
    cachingAllocator.free(ptr2);
    
    // for(auto el:v2){
    //   std::cout << el << ", ";
    // }
    std::cout << std::endl;
}