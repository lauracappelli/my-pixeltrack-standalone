#ifndef HeterogeneousCore_CUDACore_src_getCachingDeviceAllocator
#define HeterogeneousCore_CUDACore_src_getCachingDeviceAllocator

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
using namespace dpct;
#include <iomanip>
#include <iostream>

#include "CUDACore/cudaCheck.h"
#include "CUDACore/currentDevice.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/eventWorkHasCompleted.h"
#include "CUDACore/GenericCachingAllocator.h"
#include <chrono>
using namespace dpct;

namespace cms::cuda::allocator {
  // Use caching or not
  enum class Policy { Synchronous = 0, Asynchronous = 1, Caching = 2 };
#ifndef CUDADEV_DISABLE_CACHING_ALLOCATOR
  constexpr Policy policy = Policy::Caching;
#elif CUDA_VERSION >= 11020 && !defined CUDADEV_DISABLE_ASYNC_ALLOCATOR
  constexpr Policy policy = Policy::Asynchronous;
#else
  constexpr Policy policy = Policy::Synchronous;
#endif
  // Growth factor (bin_growth in cub::CachingDeviceAllocator
  constexpr unsigned int binGrowth = 2;
  // Smallest bin, corresponds to binGrowth^minBin bytes (min_bin in cub::CacingDeviceAllocator
  constexpr unsigned int minBin = 8;
  // Largest bin, corresponds to binGrowth^maxBin bytes (max_bin in cub::CachingDeviceAllocator). Note that unlike in cub, allocations larger than binGrowth^maxBin are set to fail.
  constexpr unsigned int maxBin = 30;
  // Total storage for the allocator. 0 means no limit.
  constexpr size_t maxCachedBytes = 0;
  // Fraction of total device memory taken for the allocator. In case there are multiple devices with different amounts of memory, the smallest of them is taken. If maxCachedBytes is non-zero, the smallest of them is taken.
  constexpr double maxCachedFraction = 0.8;
  constexpr bool debug = false;

  inline size_t minCachedBytes() {
    size_t ret = std::numeric_limits<size_t>::max();
    int currentDevice;
    cudaCheck(currentDevice = dev_mgr::instance().current_device_id());
    const int numberOfDevices = deviceCount();
    for (int i = 0; i < numberOfDevices; ++i) {
      size_t freeMemory, totalMemory;
      /*
      DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      cudaCheck((dev_mgr::instance().select_device(i), 0));
      /*
      DPCT1003:1: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      /*
      DPCT1072:2: DPC++ currently does not support getting the available memory on the current device. You may need to adjust the code.
      */
      cudaCheck((totalMemory = get_current_device().get_device_info().get_global_mem_size(), 0));
      ret = std::min(ret, static_cast<size_t>(maxCachedFraction * freeMemory));
    }
    /*
    DPCT1003:3: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
    cudaCheck((dev_mgr::instance().select_device(currentDevice), 0));
    if (maxCachedBytes > 0) {
      ret = std::min(ret, maxCachedBytes);
    }
    return ret;
  }

  struct DeviceTraits {
    using DeviceType = int;
    using QueueType = sycl::queue*;
    using EventType = sycl::event;

    static constexpr DeviceType kInvalidDevice = -1;

    static DeviceType currentDevice() { return cms::cuda::currentDevice(); }

    static DeviceType memoryDevice(DeviceType deviceEvent) {
      // For device allocator the device where the memory is allocated
      // on is the same as the device where the event is recorded on.
      return deviceEvent;
    }

    static bool canReuseInDevice(DeviceType a, DeviceType b) { return a == b; }

    static bool canReuseInQueue(QueueType a, QueueType b) { return a == b; }

    template <typename C>
    static bool deviceCompare(DeviceType a_dev, DeviceType b_dev, C&& compare) {
      if (a_dev == b_dev) {
        return compare();
      }
      return a_dev < b_dev;
    }

    static bool eventWorkHasCompleted(EventType e) { return cms::cuda::eventWorkHasCompleted(e); }

    static EventType createEvent() {
      EventType e;
      /*
      DPCT1027:4: The call to cudaEventCreateWithFlags was replaced with 0 because this call is redundant in DPC++.
      */
      cudaCheck(0);
      return e;
    }

    /*
    DPCT1027:5: The call to cudaEventDestroy was replaced with 0 because this call is redundant in DPC++.
    */
    static void destroyEvent(EventType e) { cudaCheck(0); }

    static EventType recreateEvent(EventType e, DeviceType prev, DeviceType next) {
      throw std::runtime_error("CUDADeviceTraits::recreateEvent() should never be called");
    }

    /*
    DPCT1012:6: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
    */
    /*
    DPCT1024:7: The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.
    */
    static void recordEvent(EventType e, QueueType queue) { e_ct1 = std::chrono::steady_clock::now();
    cudaCheck((e = queue->submit_barrier(), 0));
    std::chrono::time_point<std::chrono::steady_clock> e_ct1; }

    struct DevicePrinter {
      DevicePrinter(DeviceType device) : device_(device) {}
      void write(std::ostream& os) const { os << "Device " << device_; }
      DeviceType device_;
    };

    static DevicePrinter printDevice(DeviceType device) { return DevicePrinter(device); }

    static void* allocate(size_t bytes) {
      void* ptr;
      /*
      DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      cudaCheck((ptr = (void*)sycl::malloc_device(bytes, get_default_queue()), 0));
      return ptr;
    }

    static void* tryAllocate(size_t bytes) try {
      void* ptr;
      /*
      DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      auto error = (ptr = (void*)sycl::malloc_device(bytes, get_default_queue()), 0);

      cudaCheck(error);
      return ptr;
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    /*
    DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    */
    static void free(void* ptr) { cudaCheck((sycl::free(ptr, get_default_queue()), 0)); }
  };

  inline std::ostream& operator<<(std::ostream& os, DeviceTraits::DevicePrinter const& pr) {
    pr.write(os);
    return os;
  }

  using CachingDeviceAllocator = GenericCachingAllocator<DeviceTraits>;

  inline CachingDeviceAllocator& getCachingDeviceAllocator() {
    if (debug) {
      std::cout << "cub::CachingDeviceAllocator settings\n"
                << "  bin growth " << binGrowth << "\n"
                << "  min bin    " << minBin << "\n"
                << "  max bin    " << maxBin << "\n"
                << "  resulting bins:\n";
      for (auto bin = minBin; bin <= maxBin; ++bin) {
        auto binSize = ::allocator::intPow(binGrowth, bin);
        if (binSize >= (1 << 30) and binSize % (1 << 30) == 0) {
          std::cout << "    " << std::setw(8) << (binSize >> 30) << " GB\n";
        } else if (binSize >= (1 << 20) and binSize % (1 << 20) == 0) {
          std::cout << "    " << std::setw(8) << (binSize >> 20) << " MB\n";
        } else if (binSize >= (1 << 10) and binSize % (1 << 10) == 0) {
          std::cout << "    " << std::setw(8) << (binSize >> 10) << " kB\n";
        } else {
          std::cout << "    " << std::setw(9) << binSize << " B\n";
        }
      }
      std::cout << "  maximum amount of cached memory: " << (minCachedBytes() >> 20) << " MB\n";
    }

    // the public interface is thread safe
    static CachingDeviceAllocator allocator{binGrowth, minBin, maxBin, minCachedBytes(), debug};
    return allocator;
  }
}  // namespace cms::cuda::allocator

#endif
