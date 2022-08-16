#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
using namespace dpct;
#include <cassert>
#include <limits>

#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/allocate_device.h"
#include "CUDACore/cudaCheck.h"

#include "getCachingDeviceAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
}

namespace cms::cuda {
  void *allocate_device(int dev, size_t nbytes, sycl::queue *stream) {
    void *ptr = nullptr;
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, stream));
#if CUDA_VERSION >= 11020
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      ScopedSetDevice setDeviceForThisScope(dev);
      cudaCheck(cudaMallocAsync(&ptr, nbytes, stream));
#endif
    } else {
      ScopedSetDevice setDeviceForThisScope(dev);
      /*
      DPCT1003:30: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      cudaCheck((ptr = (void *)sycl::malloc_device(nbytes, get_default_queue()), 0));
    }
    return ptr;
  }

  void free_device(int device, void *ptr, sycl::queue *stream) {
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceFree(device, ptr));
#if CUDA_VERSION >= 11020
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      ScopedSetDevice setDeviceForThisScope(device);
      cudaCheck(cudaFreeAsync(ptr, stream));
#endif
    } else {
      ScopedSetDevice setDeviceForThisScope(device);
      /*
      DPCT1003:31: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      cudaCheck((sycl::free(ptr, get_default_queue()), 0));
    }
  }

}  // namespace cms::cuda
