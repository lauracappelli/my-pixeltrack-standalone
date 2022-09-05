#ifndef HeterogenousCore_SYCLUtilities_currentDevice_h
#define HeterogenousCore_SYCLUtilities_currentDevice_h

#include <CL/sycl.hpp>

// can be replaed by queue.get_device()
namespace cms {
  namespace sycl {
    inline sycl::device currentDevice(sycl::queue q) {
      auto dev = q.get_device();
      return dev;
    }
  }  // namespace sycl
}  // namespace cms

#endif
