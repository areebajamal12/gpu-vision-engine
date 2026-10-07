#include "gve/voxelization.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

namespace {

gve::VoxelGrid test_grid() { return {0.0F, 2.0F, 0.0F, 2.0F, 0.0F, 2.0F, 1.0F, 1.0F, 1.0F}; }

std::vector<gve::PointXYZI> boundary_points() {
  return {
      {0.0F, 0.0F, 0.0F, 0.1F},
      {0.25F, 0.50F, 0.75F, 0.2F},
      {1.0F, 1.0F, 1.0F, 0.3F},
      {1.999F, 1.999F, 1.999F, 0.4F},
      {2.0F, 1.0F, 1.0F, 0.5F},
      {-0.001F, 0.0F, 0.0F, 0.6F},
      {0.0F, 2.0F, 0.0F, 0.7F},
      {0.0F, 0.0F, 2.0F, 0.8F},
      {std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F, 0.9F},
  };
}

TEST(VoxelizationCpu, AggregatesDeterministicallyAndRejectsInvalidPoints) {
  const auto result = gve::voxelize_cpu(boundary_points(), test_grid());
  ASSERT_EQ(result.valid_point_count, 4U);
  ASSERT_EQ(result.rejected_point_count, 5U);
  ASSERT_EQ(result.voxels.size(), 2U);
  EXPECT_EQ(result.voxels[0], (gve::VoxelStats{0, 2, 250, 500, 750, 3000}));
  EXPECT_EQ(result.voxels[1], (gve::VoxelStats{7, 2, 2999, 2999, 2999, 7000}));
  EXPECT_EQ(result, gve::voxelize_cpu(boundary_points(), test_grid()));
}

TEST(VoxelizationCpu, RejectsInvalidGrid) {
  auto grid = test_grid();
  grid.voxel_x = 0.0F;
  EXPECT_THROW(gve::voxelize_cpu(boundary_points(), grid), std::invalid_argument);
}

TEST(NuScenesSweepLoader, ReadsFiveFloatRecordsAndPreservesXyzi) {
  const auto path = std::filesystem::temp_directory_path() / "gve-nuscenes-loader-test.bin";
  const std::vector<float> raw = {1.0F, 2.0F, 3.0F, 0.4F, 17.0F, -1.0F, -2.0F, -3.0F, 0.8F, 31.0F};
  {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(raw.data()),
                 static_cast<std::streamsize>(raw.size() * sizeof(float)));
  }
  const auto points = gve::load_nuscenes_lidar_sweep(path.string());
  std::filesystem::remove(path);
  ASSERT_EQ(points.size(), 2U);
  EXPECT_FLOAT_EQ(points[0].x, 1.0F);
  EXPECT_FLOAT_EQ(points[0].intensity, 0.4F);
  EXPECT_FLOAT_EQ(points[1].z, -3.0F);
  EXPECT_FLOAT_EQ(points[1].intensity, 0.8F);
}

#ifdef GVE_HAS_CUDA
TEST(VoxelizationCuda, ExactlyMatchesCpuReference) {
  const auto cpu = gve::voxelize_cpu(boundary_points(), test_grid());
  const auto cuda = gve::voxelize_cuda(boundary_points(), test_grid());
  EXPECT_EQ(cuda, cpu);
}

TEST(VoxelizationCuda, IsDeterministicAcrossRuns) {
  const auto first = gve::voxelize_cuda(boundary_points(), test_grid());
  const auto second = gve::voxelize_cuda(boundary_points(), test_grid());
  EXPECT_EQ(first, second);
}
#endif

}  // namespace
