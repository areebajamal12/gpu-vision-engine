#include "gve/voxelization.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace gve {
namespace {

struct KeyedPoint {
  std::uint32_t key;
  PointXYZI point;
};

void validate_grid(const VoxelGrid& grid) {
  if (!(grid.min_x < grid.max_x && grid.min_y < grid.max_y && grid.min_z < grid.max_z) ||
      grid.voxel_x <= 0.0F || grid.voxel_y <= 0.0F || grid.voxel_z <= 0.0F)
    throw std::invalid_argument("voxel grid bounds and sizes must be valid");
}

std::uint32_t dimension(float minimum, float maximum, float size) {
  const double cells = std::round((maximum - minimum) / size);
  if (cells <= 0.0 || cells > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("voxel grid dimension is invalid");
  return static_cast<std::uint32_t>(cells);
}

bool point_key(const PointXYZI& point, const VoxelGrid& grid, std::uint32_t nx, std::uint32_t ny,
               std::uint32_t nz, std::uint32_t* key) {
  if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z) ||
      !std::isfinite(point.intensity) || point.x < grid.min_x || point.x >= grid.max_x ||
      point.y < grid.min_y || point.y >= grid.max_y || point.z < grid.min_z ||
      point.z >= grid.max_z)
    return false;
  const auto ix = static_cast<std::uint32_t>((point.x - grid.min_x) / grid.voxel_x);
  const auto iy = static_cast<std::uint32_t>((point.y - grid.min_y) / grid.voxel_y);
  const auto iz = static_cast<std::uint32_t>((point.z - grid.min_z) / grid.voxel_z);
  if (ix >= nx || iy >= ny || iz >= nz) return false;
  *key = (iz * ny + iy) * nx + ix;
  return true;
}

std::int64_t quantize(float value, double scale) {
  return static_cast<std::int64_t>(std::llround(static_cast<double>(value) * scale));
}

}  // namespace

bool VoxelStats::operator==(const VoxelStats& other) const {
  return linear_index == other.linear_index && point_count == other.point_count &&
         sum_x_mm == other.sum_x_mm && sum_y_mm == other.sum_y_mm && sum_z_mm == other.sum_z_mm &&
         sum_intensity_1e4 == other.sum_intensity_1e4;
}

bool VoxelizationResult::operator==(const VoxelizationResult& other) const {
  return valid_point_count == other.valid_point_count &&
         rejected_point_count == other.rejected_point_count && voxels == other.voxels;
}

VoxelizationResult voxelize_cpu(const std::vector<PointXYZI>& points, const VoxelGrid& grid) {
  validate_grid(grid);
  const auto nx = dimension(grid.min_x, grid.max_x, grid.voxel_x);
  const auto ny = dimension(grid.min_y, grid.max_y, grid.voxel_y);
  const auto nz = dimension(grid.min_z, grid.max_z, grid.voxel_z);
  if (static_cast<std::uint64_t>(nx) * ny * nz >= std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("voxel grid has too many cells");

  std::vector<KeyedPoint> valid;
  valid.reserve(points.size());
  for (const auto& point : points) {
    std::uint32_t key = 0;
    if (point_key(point, grid, nx, ny, nz, &key)) valid.push_back({key, point});
  }
  std::stable_sort(valid.begin(), valid.end(),
                   [](const KeyedPoint& lhs, const KeyedPoint& rhs) { return lhs.key < rhs.key; });

  VoxelizationResult result;
  result.valid_point_count = valid.size();
  result.rejected_point_count = points.size() - valid.size();
  for (const auto& keyed : valid) {
    if (result.voxels.empty() || result.voxels.back().linear_index != keyed.key)
      result.voxels.push_back({keyed.key, 0, 0, 0, 0, 0});
    auto& voxel = result.voxels.back();
    ++voxel.point_count;
    voxel.sum_x_mm += quantize(keyed.point.x, 1000.0);
    voxel.sum_y_mm += quantize(keyed.point.y, 1000.0);
    voxel.sum_z_mm += quantize(keyed.point.z, 1000.0);
    voxel.sum_intensity_1e4 += quantize(keyed.point.intensity, 10000.0);
  }
  return result;
}

std::vector<PointXYZI> load_nuscenes_lidar_sweep(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open nuScenes LiDAR sweep: " + path);
  const auto bytes = input.tellg();
  constexpr std::streamoff record_bytes = 5 * sizeof(float);
  if (bytes <= 0 || bytes % record_bytes != 0)
    throw std::runtime_error("nuScenes LiDAR sweep must contain five float32 values per point");
  input.seekg(0);
  const auto count = static_cast<std::size_t>(bytes / record_bytes);
  std::vector<float> raw(count * 5);
  input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
  if (!input) throw std::runtime_error("failed to read nuScenes LiDAR sweep: " + path);
  std::vector<PointXYZI> points(count);
  for (std::size_t i = 0; i < count; ++i)
    points[i] = {raw[i * 5], raw[i * 5 + 1], raw[i * 5 + 2], raw[i * 5 + 3]};
  return points;
}

}  // namespace gve
