#ifndef SYCLCore_AllocateDevice_h
#define SYCLCore_AllocateDevice_h

#include <cassert>
#include <limits>

#include <CL/sycl.hpp>

#include "SYCLCore/getCachingAllocator.h"

namespace cms {
  namespace sycltools {
    // Allocate device memory
    void *allocate_device(int device, size_t nbytes, sycl::queue queue);

    // Free device memory (to be called from unique_ptr)
    void free_device(int device, void *ptr, sycl::queue queue);

  }  // namespace sycltools
}  // namespace cms

#endif