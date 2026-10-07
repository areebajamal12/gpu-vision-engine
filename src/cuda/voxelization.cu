#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "gve/voxelization.hpp"

namespace gve {
namespace {

struct Aggregate {
  std::uint32_t count;
  long long sum_x_mm;
  long long sum_y_mm;
  long long sum_z_mm;
  long long sum_intensity_1e4;
};

struct AddAggregate {
  __host__ __device__ Aggregate operator()(const Aggregate& lhs, const Aggregate& rhs) const {
    return {lhs.count + rhs.count, lhs.sum_x_mm + rhs.sum_x_mm, lhs.sum_y_mm + rhs.sum_y_mm,
            lhs.sum_z_mm + rhs.sum_z_mm, lhs.sum_intensity_1e4 + rhs.sum_intensity_1e4};
  }
};

void check_voxel(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

std::uint32_t dimension(float minimum, float maximum, float size) {
  if (!(minimum < maximum) || size <= 0.0F)
    throw std::invalid_argument("voxel grid bounds and sizes must be valid");
  const double cells = std::round((maximum - minimum) / size);
  if (cells <= 0.0 || cells > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("voxel grid dimension is invalid");
  return static_cast<std::uint32_t>(cells);
}

__global__ void map_points_kernel(const PointXYZI* points, std::uint32_t* keys, Aggregate* values,
                                  std::size_t count, VoxelGrid grid, std::uint32_t nx,
                                  std::uint32_t ny, std::uint32_t nz) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const PointXYZI point = points[index];
  const bool valid = isfinite(point.x) && isfinite(point.y) && isfinite(point.z) &&
                     isfinite(point.intensity) && point.x >= grid.min_x && point.x < grid.max_x &&
                     point.y >= grid.min_y && point.y < grid.max_y && point.z >= grid.min_z &&
                     point.z < grid.max_z;
  if (!valid) {
    keys[index] = UINT_MAX;
    values[index] = {0, 0, 0, 0, 0};
    return;
  }
  const auto ix = static_cast<std::uint32_t>((point.x - grid.min_x) / grid.voxel_x);
  const auto iy = static_cast<std::uint32_t>((point.y - grid.min_y) / grid.voxel_y);
  const auto iz = static_cast<std::uint32_t>((point.z - grid.min_z) / grid.voxel_z);
  if (ix >= nx || iy >= ny || iz >= nz) {
    keys[index] = UINT_MAX;
    values[index] = {0, 0, 0, 0, 0};
    return;
  }
  keys[index] = (iz * ny + iy) * nx + ix;
  values[index] = {1, llround(static_cast<double>(point.x) * 1000.0),
                   llround(static_cast<double>(point.y) * 1000.0),
                   llround(static_cast<double>(point.z) * 1000.0),
                   llround(static_cast<double>(point.intensity) * 10000.0)};
}

}  // namespace

VoxelizationResult voxelize_cuda(const std::vector<PointXYZI>& points, const VoxelGrid& grid,
                                 VoxelCudaTiming* timing) {
  const auto nx = dimension(grid.min_x, grid.max_x, grid.voxel_x);
  const auto ny = dimension(grid.min_y, grid.max_y, grid.voxel_y);
  const auto nz = dimension(grid.min_z, grid.max_z, grid.voxel_z);
  if (static_cast<std::uint64_t>(nx) * ny * nz >= std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("voxel grid has too many cells");
  if (points.empty()) return {};

  PointXYZI* device_points = nullptr;
  std::uint32_t* device_keys = nullptr;
  std::uint32_t* device_unique_keys = nullptr;
  Aggregate* device_values = nullptr;
  Aggregate* device_aggregates = nullptr;
  cudaEvent_t start;
  cudaEvent_t stop;
  const auto total_start = std::chrono::steady_clock::now();
  const std::size_t point_bytes = points.size() * sizeof(PointXYZI);
  const std::size_t key_bytes = points.size() * sizeof(std::uint32_t);
  const std::size_t value_bytes = points.size() * sizeof(Aggregate);
  check_voxel(cudaMalloc(&device_points, point_bytes), "cudaMalloc points");
  check_voxel(cudaMalloc(&device_keys, key_bytes), "cudaMalloc keys");
  check_voxel(cudaMalloc(&device_unique_keys, key_bytes), "cudaMalloc unique keys");
  check_voxel(cudaMalloc(&device_values, value_bytes), "cudaMalloc values");
  check_voxel(cudaMalloc(&device_aggregates, value_bytes), "cudaMalloc aggregates");
  check_voxel(cudaMemcpy(device_points, points.data(), point_bytes, cudaMemcpyHostToDevice),
              "copy points");
  check_voxel(cudaEventCreate(&start), "create start event");
  check_voxel(cudaEventCreate(&stop), "create stop event");
  check_voxel(cudaEventRecord(start), "record start");

  constexpr int threads = 256;
  const int blocks = static_cast<int>((points.size() + threads - 1) / threads);
  map_points_kernel<<<blocks, threads>>>(device_points, device_keys, device_values, points.size(),
                                         grid, nx, ny, nz);
  check_voxel(cudaGetLastError(), "launch point mapping kernel");
  auto keys_begin = thrust::device_pointer_cast(device_keys);
  auto values_begin = thrust::device_pointer_cast(device_values);
  thrust::stable_sort_by_key(thrust::device, keys_begin, keys_begin + points.size(), values_begin);
  auto reduced =
      thrust::reduce_by_key(thrust::device, keys_begin, keys_begin + points.size(), values_begin,
                            thrust::device_pointer_cast(device_unique_keys),
                            thrust::device_pointer_cast(device_aggregates),
                            thrust::equal_to<std::uint32_t>(), AddAggregate{});
  const std::size_t reduced_count = reduced.first - thrust::device_pointer_cast(device_unique_keys);
  check_voxel(cudaEventRecord(stop), "record stop");
  check_voxel(cudaEventSynchronize(stop), "synchronize voxelization");
  float kernel_ms = 0.0F;
  check_voxel(cudaEventElapsedTime(&kernel_ms, start, stop), "measure voxelization");

  std::vector<std::uint32_t> keys(reduced_count);
  std::vector<Aggregate> aggregates(reduced_count);
  check_voxel(cudaMemcpy(keys.data(), device_unique_keys, reduced_count * sizeof(std::uint32_t),
                         cudaMemcpyDeviceToHost),
              "copy voxel keys");
  check_voxel(cudaMemcpy(aggregates.data(), device_aggregates, reduced_count * sizeof(Aggregate),
                         cudaMemcpyDeviceToHost),
              "copy voxel aggregates");

  VoxelizationResult result;
  const std::size_t occupied = reduced_count - (keys.back() == UINT_MAX ? 1 : 0);
  result.voxels.reserve(occupied);
  for (std::size_t i = 0; i < occupied; ++i) {
    const auto& value = aggregates[i];
    result.voxels.push_back({keys[i], value.count, value.sum_x_mm, value.sum_y_mm, value.sum_z_mm,
                             value.sum_intensity_1e4});
    result.valid_point_count += value.count;
  }
  result.rejected_point_count = points.size() - result.valid_point_count;
  const auto total_stop = std::chrono::steady_clock::now();
  if (timing) {
    timing->kernel_ms = kernel_ms;
    timing->end_to_end_ms =
        std::chrono::duration<double, std::milli>(total_stop - total_start).count();
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_points);
  cudaFree(device_keys);
  cudaFree(device_unique_keys);
  cudaFree(device_values);
  cudaFree(device_aggregates);
  return result;
}

}  // namespace gve
