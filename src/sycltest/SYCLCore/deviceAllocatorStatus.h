#ifndef HeterogeneousCore_SYCLtilities_deviceAllocatorStatus_h
#define HeterogeneousCore_SYCLtilities_deviceAllocatorStatus_h

#include <cstddef>
#include <map>

namespace cms {
  namespace sycl {
    namespace allocator {
      struct TotalBytes {
        size_t free = 0;
        size_t live = 0;
        size_t liveRequested = 0;
      };
      /// Map type of device ordinals to the number of cached bytes cached by each device
      using DeviceCachedBytes = std::map<int, TotalBytes>;
    }  // namespace allocator

    allocator::DeviceCachedBytes deviceAllocatorStatus();
  }  // namespace sycl
}  // namespace cms

#endif
