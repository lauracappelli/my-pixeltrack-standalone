#include "SYCLCore/deviceAllocatorStatus.h"

#include "getCachingDeviceAllocator.h"

namespace cms::sycl {
  allocator::DeviceCachedBytes deviceAllocatorStatus() { return allocator::getCachingDeviceAllocator().cacheStatus(); }
}  // namespace cms::sycl
