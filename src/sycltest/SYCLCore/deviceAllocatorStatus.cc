#include "SYCLCore/deviceAllocatorStatus.h"

#include "getCachingDeviceAllocator.h"

namespace cms::sycl {
  allocator::GpuCachedBytes deviceAllocatorStatus() { return allocator::getCachingDeviceAllocator().cacheStatus(); }
}  // namespace cms::sycl
