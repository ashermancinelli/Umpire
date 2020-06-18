
#include <iostream>
#include "umpire/ResourceManager.hpp"

constexpr std::size_t N = 100;
constexpr double value = 3.;

__global__
void verify_data(double* data)
{
  int tid = threadIdx.x;
  if(tid < N)
  {
    assert(data[tid] == value);
  }
}

/**
 *
 * This function may recieve a host-side pointer from any other library,
 * register it with the resource manager, and copy it to/from devices
 *
 */
void use_external_pointer(double* external_source)
{
  auto& resmgr = umpire::ResourceManager::getInstance();
  auto host_allocator = resmgr.getAllocator("HOST");
  auto device_allocator = resmgr.getAllocator("DEVICE");

  auto device_ptr = static_cast<double*>(device_allocator.allocate(sizeof(double) * N));

  /*
   * If you expect the incoming pointer to already exist on the device,
   * then register the source with the device_allocator's
   * allocation strategy instead of the host's as we do here.
   */
  umpire::strategy::AllocationStrategy* strategy = host_allocator.getAllocationStrategy();
  umpire::util::AllocationRecord record{external_source, sizeof(double)*N, strategy};
  resmgr.registerAllocation(external_source, record);
  resmgr.copy(device_ptr, external_source);

  // Ensure that the host data was copied over to device
  verify_data<<<1, N>>>(device_ptr);

  resmgr.deallocate(device_ptr);
}

int main(int argc, char** argv)
{
  // Initialize data on host
  auto data = new double[N];
  for(int i=0; i<N; i++) data[i] = value;

  use_external_pointer(data);

  delete[] data;
}
