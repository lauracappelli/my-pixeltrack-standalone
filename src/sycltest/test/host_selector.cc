#include <CL/sycl.hpp>

int main() {
  auto d = sycl::device(sycl::host_selector());
  std::cout << "Selected device: " << d.get_info<sycl::info::device::name>() << "\n";
  if(d.is_host()){
    std::cout << "The device is the host!\n";
  }

  auto hd = sycl::device::get_devices(sycl::info::device_type::host);
  std::cout << "host vector size: " << hd.size() << "\n";
  std::cout << "Selected device: " << hd[0].get_info<sycl::info::device::name>() << "\n";
  if(hd[0].is_host()){
    std::cout << "The device is the host!\n";
  }
}