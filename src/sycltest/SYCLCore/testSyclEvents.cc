#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <random>

void random_fill(std::vector<int>& v)
{
    std::random_device r;
    std::mt19937 gen(r());
    for (auto i = v.begin(); i < v.end(); i++) {
        *i = gen()%100;
    }
}

void sequenzial_sum(std::vector<int>& a, std::vector<int>& b, std::vector<int>& c)
{
    for (auto i = 0; i < c.size(); i++) {
        c[i] = a[i] + b[i];
    }
}

void parallel_sum(const int *a, const int *b, int *c, sycl::nd_item<3> item)
{
    auto i = item.get_local_id(2) +
             item.get_group(2) * item.get_local_range().get(2);
    c[i] = a[i] + b[i];
}

bool check_sum(std::vector<int>& a, std::vector<int>& b)
{
  bool flag = true;
  int i = 0;
  while (flag && i < a.size()) {
    flag = a[i] == b[i];
    i++;
  }
  return flag;
}

int main() {
  
  // create the default queue
  sycl::queue q;
  std::cout << "\nPlatform: " << q.get_device().get_platform().get_info<sycl::info::platform::name>() << '\n';
  
  // create "empty" event
  sycl::event empty_event;

  constexpr int N = 16392 * 16392;
  constexpr int size = N * sizeof(int);
  constexpr int threads_per_block = 1024;

  // vector inizialization
  std::vector<int> h_vec_a, h_vec_b, h_vec_c, h_vec_d;
  int *d_vec_a, *d_vec_b, *d_vec_c;
  h_vec_a.resize(N);
  h_vec_b.resize(N);
  h_vec_c.resize(N);
  h_vec_d.resize(N);
  random_fill(h_vec_a);
  random_fill(h_vec_b);

  // host vector sum
  sequenzial_sum(h_vec_a, h_vec_b, h_vec_d);

  // Allocate space in device memory
  d_vec_a = (int *)sycl::malloc_device(size, q);
  d_vec_b = (int *)sycl::malloc_device(size, q);
  d_vec_c = (int *)sycl::malloc_device(size, q);

  // copy data from host to device
  sycl::event memcpy_event_vec_a = q.memcpy(d_vec_a, h_vec_a.data(), size);
  sycl::event memcpy_event_vec_b = q.memcpy(d_vec_b, h_vec_b.data(), size);

  // vector sum kernel
  sycl::event sum_event = q.submit([&](sycl::handler &h) {
    h.depends_on(memcpy_event_vec_a);
    h.depends_on(memcpy_event_vec_b);
    h.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, N / threads_per_block) *
      sycl::range<3>(1, 1, threads_per_block), sycl::range<3>(1, 1, threads_per_block)),
    [=](sycl::nd_item<3> item) {
      parallel_sum(d_vec_a, d_vec_b, d_vec_c, item);
    });
  });

  // vector of events that sum_event waits on
  auto event_list = sum_event.get_wait_list();
  std::cout << "\t- Size of the event list of sum_event = " << event_list.size() << '\n';

  // Wait for the associated command to complete
  sum_event.wait();

  // copy result from device to host and wait for completition passing asynchronous errors to the handler
  sycl::event memcpy_event_vec_c = q.memcpy(h_vec_c.data(), d_vec_c, size);
  memcpy_event_vec_c.wait_and_throw();
  // auto event_list = sum_event.get_wait_list();
  // q.memcpy(h_vec_c.data(), d_vec_c, size).wait_and_throw(event_list.push_back(sum_event));

  sycl::free(d_vec_a, q);
  sycl::free(d_vec_b, q);
  sycl::free(d_vec_c, q);

  if (check_sum(h_vec_c, h_vec_d))
    std::cout << "\nVector sum correct\n\n";
  else
    std::cout << "\nSomething went wrong in vector sum\n\n";


  /* 
  * SYCL EVENTS
  */
  std::cout << "EVENT INFORMATIONS\n";

  // vector of events that sum_event waits on
  // auto event_list = sum_event.get_wait_list();
  // std::cout << "\t- Size of the event list of sum_event = " << event_list.size() << '\n';

  // Reference count of the event
  auto ref_count = sum_event.get_info<sycl::info::event::reference_count>();
  std::cout << "\t- sum_event reference count = " << ref_count << "\n";

  // Ask if events are host event
  std::cout << "\t- The empty_event ";
  if (empty_event.is_host())
    std::cout << "is host event\n";
  else
    std::cout << "is not host event\n";

  // Command execution status. It could be submitted, running or complete
  if (empty_event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::submitted)
    std::cout << "\t- The empty_event is submitted\n";
  else if (empty_event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::running)
    std::cout << "\t- The empty_event is running\n";
  else if (empty_event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete)
    std::cout << "\t- The empty_event is complete\n";

  std::cout << "\t- sum_event execution time: \n";
  // get_profiling_info
  // Time in nanoseconds when sycl::command_group was submitted
  auto submit = sum_event.get_profiling_info<sycl::info::event_profiling::command_submit>();
  std::cout << "\t  Submit time\t" << submit / 1.0e9 << " seconds\n";
  // Time in nanoseconds when sycl::command_group started execution
  auto start  = sum_event.get_profiling_info<sycl::info::event_profiling::command_start>();
  std::cout << "\t  Start time\t" << start / 1.0e9 << " seconds\n";
  // Time in nanoseconds when sycl::command_group finished execution
  auto end    = sum_event.get_profiling_info<sycl::info::event_profiling::command_end>();
  std::cout << "\t  End time\t" << end / 1.0e9 << " seconds\n";
  // print results
  std::cout << "\t  Kernel execution time = " << (end - start) / 1.0e9 << "\n";
  std::cout << "\t  Total command group processing time = " << (end - submit) / 1.0e9 << "\n\n";

}