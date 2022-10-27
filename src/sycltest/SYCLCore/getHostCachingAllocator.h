#ifndef SYCLCore_getHostCachingAllocator_h
#define SYCLCore_getHostCachingAllocator_h

#include "SYCLCore/AllocatorConfig.h"
#include "SYCLCore/CachingAllocator.h"

namespace cms::sycltools {

  inline CachingAllocator& getHostCachingAllocator() {
    static sycl::device host = sycl::device::get_devices(sycl::info::device_type::host)[0];
    // thread safe initialisation of the host allocator
    static CachingAllocator allocator(host,
                                      config::binGrowth,
                                      config::minBin,
                                      config::maxBin,
                                      config::maxCachedBytes,
                                      config::maxCachedFraction,
                                      config::allocator_policy,  // reuseSameQueueAllocations
                                      false);                    // debug

    // the public interface is thread safe
    return allocator;
  }

}  // namespace cms::sycltools

#endif  // SYCLCore_getHostCachingAllocator_h
