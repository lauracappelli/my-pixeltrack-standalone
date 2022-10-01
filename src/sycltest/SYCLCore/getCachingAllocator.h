#ifndef SYCLCore_getCachingAllocator_h
#define SYCLCore_getCachingAllocator_h

#include <optional>
#include <mutex>
#include <vector>

#include <CL/sycl.hpp>

#include "SYCLCore/getDeviceIndex.h"
#include "SYCLCore/AllocatorConfig.h"
#include "SYCLCore/CachingAllocator.h"

namespace cms::sycltools {

  namespace detail {
    inline auto allocate_allocators() {
      using Allocator = cms::sycltools::CachingAllocator;
      auto const& devices = enumerateDevices();
      auto const size = devices.size();

      // allocate the storage for the objects
      auto ptr = std::allocator<Allocator>().allocate(size);

      // construct the objects in the storage
      for (size_t index = 0; index < size; ++index) {
        new (ptr + index) Allocator(devices[index],
                                    config::binGrowth,
                                    config::minBin,
                                    config::maxBin,
                                    config::maxCachedBytes,
                                    config::maxCachedFraction,
                                    config::allocator_policy,  // reuseSameQueueAllocations
                                    false);                    // debug
      }

      // use a custom deleter to destroy all objects and deallocate the memory
      auto deleter = [size](Allocator* ptr) {
        for (size_t i = size; i > 0; --i) {
          (ptr + i - 1)->~Allocator();
        }
        std::allocator<Allocator>().deallocate(ptr, size);
      };

      return std::unique_ptr<Allocator[], decltype(deleter)>(ptr, deleter);
    }

  }  // namespace detail

  inline CachingAllocator& getCachingAllocator(sycl::device const& device) {
    // initialise all allocators, one per device
    static auto allocators = detail::allocate_allocators();

    size_t const index = getDeviceIndex(device);
    assert(index < cms::sycltools::enumerateDevices().size());

    // the public interface is thread safe
    return allocators[index];
  }

}  // namespace cms::sycltools

#endif  // SYCLCore_getDeviceCachingAllocator_h
