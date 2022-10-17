#ifndef HeterogeneousCore_SYCLUtilities_interface_unique_ptr_h
#define HeterogeneousCore_SYCLUtilities_interface_unique_ptr_h

#include <functional>
#include <memory>
#include <optional>

#include <CL/sycl.hpp>
#include "SYCLCore/getCachingAllocator.h"

namespace cms {
  namespace sycltools {
    namespace impl {
      // Additional layer of types to distinguish from host::unique_ptr
      class Deleter {
      public:
        Deleter() = default;  // for edm::Wrapper
        Deleter(sycl::queue stream) : stream_{stream} {}

        void operator()(void* ptr) {
          if (stream_) {
            auto dev = (*stream_).get_device();
            CachingAllocator& allocator = getCachingAllocator(dev);
            allocator.free(ptr);
          }
        }

      private:
        std::optional<sycl::queue> stream_;
      };
    }  // namespace impl

    template <typename T>
    using unique_ptr = std::unique_ptr<T, impl::Deleter>;

    namespace impl {
      template <typename T>
      struct make_unique_selector {
        using non_array = cms::sycltools::unique_ptr<T>;
      };
      template <typename T>
      struct make_unique_selector<T[]> {
        using unbounded_array = cms::sycltools::unique_ptr<T[]>;
      };
      template <typename T, size_t N>
      struct make_unique_selector<T[N]> {
        struct bounded_array {};
      };
    }  // namespace impl

    template <typename T>
    typename impl::make_unique_selector<T>::non_array make_unique(sycl::queue const& stream) {
      static_assert(std::is_trivially_constructible<T>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      CachingAllocator& allocator = getCachingAllocator(stream.get_device());
      void* mem = allocator.allocate(sizeof(T), stream);
      return typename impl::make_unique_selector<T>::non_array{reinterpret_cast<T*>(mem), impl::Deleter{stream}};
    }

    template <typename T>
    typename impl::make_unique_selector<T>::unbounded_array make_unique(size_t n, sycl::queue const& stream) {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_constructible<element_type>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      CachingAllocator& allocator = getCachingAllocator(stream.get_device());
      void* mem = allocator.allocate(n * sizeof(element_type), stream);
      return typename impl::make_unique_selector<T>::unbounded_array{reinterpret_cast<element_type*>(mem),
                                                                     impl::Deleter{stream}};
    }

    template <typename T, typename... Args>
    typename impl::make_unique_selector<T>::bounded_array make_unique(Args&&...) = delete;

    // No check for the trivial constructor, make it clear in the interface
    template <typename T>
    typename impl::make_unique_selector<T>::non_array make_unique_uninitialized(sycl::queue const& stream) {
      CachingAllocator& allocator = getCachingAllocator(stream.get_device());
      void* mem = allocator.allocate(sizeof(T), stream);
      return typename impl::make_unique_selector<T>::non_array{reinterpret_cast<T*>(mem), impl::Deleter{stream}};
    }

    template <typename T>
    typename impl::make_unique_selector<T>::unbounded_array make_unique_uninitialized(size_t n, sycl::queue const& stream) {
      using element_type = typename std::remove_extent<T>::type;
      CachingAllocator& allocator = getCachingAllocator(stream.get_device());
      void* mem = allocator.allocate(n * sizeof(element_type), stream);
      return typename impl::make_unique_selector<T>::unbounded_array{reinterpret_cast<element_type*>(mem),
                                                                     impl::Deleter{stream}};
    }

    template <typename T, typename... Args>
    typename impl::make_unique_selector<T>::bounded_array make_unique_uninitialized(Args&&...) = delete;
  }  // namespace sycltools
}  // namespace cms

#endif //HeterogeneousCore_SYCLUtilities_interface_unique_ptr_h
