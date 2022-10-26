#include <CL/sycl.hpp>
#include "SYCLCore/host_unique_ptr.h"
#include "SYCLCore/device_unique_ptr.h"

#include <vector>

namespace cms::sycltools {
  std::vector<sycl::device> const& discoverDevices() {
    static std::vector<sycl::device> temp;
    std::vector<sycl::device> cpus = sycl::device::get_devices(sycl::info::device_type::cpu);
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    std::vector<sycl::device> hosts = sycl::device::get_devices(sycl::info::device_type::host);
    for (auto it = cpus.begin(); it != cpus.end(); it++) {
      if (it + 1 == cpus.end()) {
        break;
      }
      if ((*it).get_info<sycl::info::device::name>() == (*(it + 1)).get_info<sycl::info::device::name>() and
          (*it).get_backend() == (*(it + 1)).get_backend() and
          (*(it + 1)).get_info<sycl::info::device::driver_version>() <
              (*it).get_info<sycl::info::device::driver_version>()) {
        cpus.erase(it + 1);
      }
    }
    temp.insert(temp.end(), cpus.begin(), cpus.end());

    for (auto it = gpus.begin(); it != gpus.end(); it++) {
      if (it + 1 == gpus.end()) {
        break;
      }
      if ((*it).get_info<sycl::info::device::name>() == (*(it + 1)).get_info<sycl::info::device::name>() and
          (*it).get_backend() == (*(it + 1)).get_backend() and
          (*(it + 1)).get_info<sycl::info::device::driver_version>() <
              (*it).get_info<sycl::info::device::driver_version>()) {
        gpus.erase(it + 1);
      }
    }
    temp.insert(temp.end(), gpus.begin(), gpus.end());
    temp.insert(temp.end(), hosts.begin(), hosts.end());
    return temp;
  }

  std::vector<sycl::device> const& enumerateDevices(bool verbose) {
    static const std::vector<sycl::device> devices = discoverDevices();

    if (verbose) {
      std::cerr << "Found " << devices.size() << " SYCL devices:" << std::endl;
      for (auto const& device : devices)
        std::cerr << "  - " << device.get_backend() << ' ' << device.get_info<cl::sycl::info::device::name>() << " ["
                  << device.get_info<sycl::info::device::driver_version>() << "]" << std::endl;
      std::cerr << std::endl;
    }
    return devices;
  }

  sycl::device chooseDevice(int id, bool debug) {
    auto const& devices = cms::sycltools::enumerateDevices(debug);
    auto const& device = devices[id % devices.size()];
    if (debug) {
      std::cerr << "Stream ID " << id << " offload to " << device.get_info<cl::sycl::info::device::name>()
                << " on backend " << device.get_backend() << std::endl;
    }
    return device;
  }
}  // namespace cms::sycltools

int main() {

  // define host and device sycl::device and the corresponing queues
  sycl::device host = cms::sycltools::chooseDevice(3, true);
  sycl::device device = cms::sycltools::chooseDevice(2, true);
  sycl::queue host_queue = sycl::queue{sycl::cpu_selector()};
  sycl::queue dev_queue = sycl::queue{sycl::gpu_selector()};

  unsigned int N = 1000;
  std::vector<int> vec(N);
  for (auto i=0; i < N; i++) {
    vec[i] = i;
  }

  auto ptr1 = cms::sycltools::make_host_unique<int[]>(N * sizeof(int), host_queue);
  auto ptr2 = cms::sycltools::make_host_unique<int[]>(N * sizeof(int), dev_queue);
  auto ptr3 = cms::sycltools::make_device_unique<int[]>(N * sizeof(int), host_queue);
  auto ptr4 = cms::sycltools::make_device_unique<int[]>(N * sizeof(int), dev_queue);

  // ptr1
  int* int_ptr1 = ptr1.get();
  host_queue.memcpy(vec.data(), int_ptr1, N * sizeof(int));
  host_queue.wait();

  for (auto i=0; i < N; i++) {
    int_ptr1[i] = 42;
  }

  // ptr2
  int* int_ptr2 = ptr2.get();
  dev_queue.memcpy(vec.data(), int_ptr2, N * sizeof(int));
  dev_queue.wait();

  for (auto i=0; i < N; i++) {
    int_ptr2[i] = 42;
  }

  // ptr3
  int* int_ptr3 = ptr3.get();
  host_queue.memcpy(vec.data(), int_ptr3, N * sizeof(int));
  host_queue.wait();

  // ptr4
  int* int_ptr4 = ptr4.get();
  dev_queue.memcpy(vec.data(), int_ptr4, N * sizeof(int));
  dev_queue.wait();

  // try this to seg fault
  // for (auto i=0; i < N; i++) {
  //   int_ptr4[i] = 42;
  // }

}