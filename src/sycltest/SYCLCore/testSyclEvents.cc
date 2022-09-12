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

void single_add(const int *a, const int *b, int *c)
{
    *c = *a + *b;
}

int main() {
  
  constexpr int N = 16 * 16;
  constexpr int size = N * sizeof(int);
  constexpr int threads_per_block = 16;

  // create the default queue
  sycl::queue q;
  std::cout << "Platform: " << q.get_device().get_platform().get_info<sycl::info::platform::name>() << '\n';
  
  // create "empty" event
  sycl::event empty_event;

  // simple sum variables
  int h_a = 2, h_b = 7, h_c;
  int *d_a, *d_b, *d_c;

  // vector sum variables
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
  d_a = (int *)sycl::malloc_device(sizeof(int), q);
  d_b = (int *)sycl::malloc_device(sizeof(int), q);
  d_c = (int *)sycl::malloc_device(sizeof(int), q);
  d_vec_a = (int *)sycl::malloc_device(size, q);
  d_vec_b = (int *)sycl::malloc_device(size, q);
  d_vec_c = (int *)sycl::malloc_device(size, q);

  // copy data from host to device
  sycl::event memcpy_event1 = q.memcpy(d_a, &h_a, sizeof(int));
  sycl::event memcpy_event2 = q.memcpy(d_b, &h_b, sizeof(int));
  sycl::event memcpy_event4 = q.memcpy(d_vec_a, h_vec_a.data(), size);
  sycl::event memcpy_event5 = q.memcpy(d_vec_b, h_vec_b.data(), size);

  // simple sum kernel
  sycl::event device_event = q.submit([&](sycl::handler &h) {
    h.depends_on(memcpy_event1);
    h.depends_on(memcpy_event2);
    h.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item) {
          single_add(d_a, d_b, d_c);
      });
  });

  // vector sum kernel
  sycl::event device_event2 = q.submit([&](sycl::handler &h) {
    h.depends_on(memcpy_event4);
    h.depends_on(memcpy_event5);
    h.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, N / threads_per_block) *
      sycl::range<3>(1, 1, threads_per_block), sycl::range<3>(1, 1, threads_per_block)),
    [=](sycl::nd_item<3> item) {
      parallel_sum(d_vec_a, d_vec_b, d_vec_c, item);
    });
  });

  // vector of events that this events waits on
  auto event_list = device_event.get_wait_list();
  std::cout << "Size of the event list of device_event " << event_list.size() << '\n';

  // copy result from device to host
  // Wait for the associated command to complete
  device_event.wait();
  device_event2.wait();
  sycl::event memcpy_event3 = q.memcpy(&h_c, d_c, sizeof(int));
  sycl::event memcpy_event6 = q.memcpy(h_vec_c.data(), d_vec_c, size);
  // Wait for an event to complete, and pass asynchronous errors to handler associated with the command
  memcpy_event3.wait_and_throw();
  memcpy_event6.wait_and_throw();

  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_c, q);
  sycl::free(d_vec_a, q);
  sycl::free(d_vec_b, q);
  sycl::free(d_vec_c, q);

  // Ask if events are host event
  std::cout << "empty_event ";
  if (empty_event.is_host())
    std::cout << "is host event\n";
  else
    std::cout << "is not host event\n";

  std::cout << "memcpy_event ";
  if (memcpy_event1.is_host())
    std::cout << "is host event\n";
  else
    std::cout << "is not host event\n";
 
  // // Wait for vector of events to complete
  // e.wait(device_event.get_wait_list());
  // // Wait for a vector of events to complete, and pass asynchronous errors to handlers associated with the commands
  // e.wait_and_throw(device_event.get_wait_list());

  /*
   get_info
  */
  // Reference count of the event
  auto ref_count = device_event.get_info<sycl::info::event::reference_count>();
  std::cout << "device_event reference count " << ref_count << "\n";
  // Command execution status. It could be submitted, running or complete
  if (empty_event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::submitted)
    std::cout << "The empty_event is submitted\n";
  else if (empty_event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::running)
    std::cout << "The empty_event is running\n";
  else if (empty_event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete)
    std::cout << "The empty_event is complete\n";

  /*
    get_profiling_info
  */
  // Time in nanoseconds when sycl::command_group was submitted
  auto submit = device_event2.get_profiling_info<sycl::info::event_profiling::command_submit>();
  std::cout << "Submit time " << submit / 1.0e9 << " seconds\n";
  // Time in nanoseconds when sycl::command_group started execution
  auto start  = device_event2.get_profiling_info<sycl::info::event_profiling::command_start>();
  std::cout << "Start time " << start / 1.0e9 << " seconds\n";
  // Time in nanoseconds when sycl::command_group finished execution
  auto end    = device_event2.get_profiling_info<sycl::info::event_profiling::command_end>();
  std::cout << "End time " << end / 1.0e9 << " seconds\n";

  std::cout << "Kernel execution time = " << (end - start) / 1.0e9 << "\n";
  std::cout << "Total command group processing time = " << (end - submit) / 1.0e9 << "\n";

  std::cout << "Simple sum - Host result: " << h_a + h_b << ". Device result: " << h_c <<  "\n";
  if (check_sum(h_vec_c, h_vec_d))
    std::cout << "Vector sum correct\n";
}