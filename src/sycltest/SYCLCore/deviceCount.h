#ifndef HeterogenousCore_SYCLUtilities_deviceCount_h
#define HeterogenousCore_SYCLUtilities_deviceCount_h

#include <CL/sycl.hpp>
#include "SYCLCore/chooseDevice.h"

// this function could be deleted
namespace cms {
  namespace sycl {
    inline int deviceCount() {
      return enumerateDevices().size();
    }
  }  // namespace sycl
}  // namespace cms

#endif
