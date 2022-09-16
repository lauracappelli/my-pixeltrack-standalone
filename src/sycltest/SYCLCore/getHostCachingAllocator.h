#ifndef SYCLCore_getHostCachingAllocator_h
#define SYCLCore_getHostCachingAllocator_h

#include "SYCLCore/AllocatorConfig.h"
#include "SYCLCore/CachingAllocator.h"

namespace cms::sycltools {
  
  template <typename TDevice, typename TQueue>
    inline CachingAllocator<TDevice, TQueue>& getHostCachingAllocator() {
      // thread safe initialisation of the host allocator
      static CachingAllocator<TDevice, TQueue> allocator (host,
                                                          config::binGrowth,
                                                          config::minBin,
                                                          config::maxBin,
                                                          config::maxCachedBytes,
                                                          config::maxCachedFraction,
                                                          false,   // reuseSameQueueAllocations
                                                          false);  // debug

      // the public interface is thread safe
      return allocator;
    }

}  // namespace cms::sycltools

#endif  // SYCLCore_getHostCachingAllocator_h
