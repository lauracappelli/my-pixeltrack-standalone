#ifndef HeterogenousCore_CUDAUtilities_currentDevice_h
#define HeterogenousCore_CUDAUtilities_currentDevice_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
using namespace dpct;
#include "CUDACore/cudaCheck.h"

namespace cms {
  namespace cuda {
    inline int currentDevice() {
      int dev;
      cudaCheck(dev = dev_mgr::instance().current_device_id());
      return dev;
    }
  }  // namespace cuda
}  // namespace cms

#endif
