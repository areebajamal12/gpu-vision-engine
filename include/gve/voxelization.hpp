#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gve {

struct PointXYZI {
  float x;
  float y;
  float z;
  float intensity;
};

struct VoxelGrid {
  float min_x{-50.0F};
  float max_x{50.0F};
  float min_y{-50.0F};
  float max_y{50.0F};
  float min_z{-5.0F};
  float max_z{3.0F};
  float voxel_x{0.25F};
  float voxel_y{0.25F};
  float voxel_z{0.20F};
};

struct VoxelStats {
  std::uint32_t linear_index;
  std::uint32_t point_count;
  std::int64_t sum_x_mm;
  std::int64_t sum_y_mm;
  std::int64_t sum_z_mm;
  std::int64_t sum_intensity_1e4;

  bool operator==(const VoxelStats& other) const;
};

struct VoxelizationResult {
  std::vector<VoxelStats> voxels;
  std::uint64_t valid_point_count{};
  std::uint64_t rejected_point_count{};

  bool operator==(const VoxelizationResult& other) const;
};

struct VoxelCudaTiming {
  double kernel_ms{};
  double end_to_end_ms{};
};

VoxelizationResult voxelize_cpu(const std::vector<PointXYZI>& points, const VoxelGrid& grid);
std::vector<PointXYZI> load_nuscenes_lidar_sweep(const std::string& path);

#ifdef GVE_HAS_CUDA
VoxelizationResult voxelize_cuda(const std::vector<PointXYZI>& points, const VoxelGrid& grid,
                                 VoxelCudaTiming* timing = nullptr);
#endif

}  // namespace gve
