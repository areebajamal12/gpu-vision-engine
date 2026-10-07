#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "environment.hpp"
#include "gve/voxelization.hpp"

#ifndef GVE_COMPILER_FLAGS
#define GVE_COMPILER_FLAGS "CMake build-type flags; see CMakeCache.txt"
#endif
#ifndef GVE_CUDA_COMPILER
#define GVE_CUDA_COMPILER "not enabled"
#define GVE_CUDA_FLAGS "not enabled"
#endif

namespace {
using Clock = std::chrono::steady_clock;

struct Statistics {
  double mean;
  double p95;
  double million_points_per_second;
};

struct Result {
  std::string backend;
  std::string scope;
  Statistics statistics;
  double mean_speedup_vs_cpu;
  double p95_speedup_vs_cpu;
};

Statistics summarize(std::vector<double> values, std::size_t point_count) {
  std::sort(values.begin(), values.end());
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  const auto p95_index = static_cast<std::size_t>(std::ceil(values.size() * 0.95)) - 1;
  return {mean, values[p95_index], static_cast<double>(point_count) / 1e6 / (mean / 1000.0)};
}

template <typename Function>
std::vector<double> measure_cpu(Function&& function, int warmups, int iterations) {
  for (int i = 0; i < warmups; ++i) function();
  std::vector<double> samples;
  samples.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    const auto start = Clock::now();
    function();
    const auto stop = Clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
  }
  return samples;
}

std::string environment_value(const char* name, const std::string& fallback) {
  const char* value = std::getenv(name);
  return value && *value ? value : fallback;
}

void write_json(const std::string& path, const std::string& sweep_path,
                const std::vector<gve::PointXYZI>& points, const gve::VoxelGrid& grid,
                const gve::VoxelizationResult& reference, const std::vector<Result>& results,
                int warmups, int iterations) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot open output file: " + path);
  output << std::fixed << std::setprecision(6);
  output << "{\n  \"schema_version\": 1,\n  \"correctness_gate_passed\": true,\n";
  output << "  \"dataset\": {\"name\": \"nuScenes\", \"sample_type\": \"real_lidar_sweep\", "
            "\"file\": \""
         << gve::benchmark::json_escape(std::filesystem::path(sweep_path).filename().string())
         << "\", \"sha256\": \""
         << gve::benchmark::json_escape(environment_value("NUSCENES_SWEEP_SHA256", "unavailable"))
         << "\"},\n";
  output << "  \"environment\": {\n";
  output << "    \"cpu_model\": \"" << gve::benchmark::json_escape(gve::benchmark::cpu_model())
         << "\",\n";
  output << "    \"gpu_model\": \"" << gve::benchmark::json_escape(gve::benchmark::gpu_model())
         << "\",\n";
  output << "    \"cuda_runtime_version\": \"" << gve::benchmark::cuda_runtime_version() << "\",\n";
  output << "    \"cuda_toolkit_version\": \"" << gve::benchmark::cuda_toolkit_version() << "\",\n";
  output << "    \"cuda_driver_version\": \"" << gve::benchmark::cuda_driver_version() << "\",\n";
  output << "    \"compiler\": \"" << gve::benchmark::json_escape(__VERSION__) << "\",\n";
  output << "    \"compiler_flags\": \"" << gve::benchmark::json_escape(GVE_COMPILER_FLAGS)
         << "\",\n";
  output << "    \"cuda_compiler\": \"" << gve::benchmark::json_escape(GVE_CUDA_COMPILER)
         << "\",\n";
  output << "    \"cuda_compiler_flags\": \"" << gve::benchmark::json_escape(GVE_CUDA_FLAGS)
         << "\"\n  },\n";
  output << "  \"workload\": {\"operation\": \"lidar_voxelization\", \"input_points\": "
         << points.size() << ", \"valid_points\": " << reference.valid_point_count
         << ", \"rejected_points\": " << reference.rejected_point_count
         << ", \"occupied_voxels\": " << reference.voxels.size()
         << ", \"point_fields\": [\"x\", \"y\", \"z\", \"intensity\"], "
            "\"bounds\": {\"x\": ["
         << grid.min_x << ", " << grid.max_x << "], \"y\": [" << grid.min_y << ", " << grid.max_y
         << "], \"z\": [" << grid.min_z << ", " << grid.max_z << "]}, \"voxel_size\": ["
         << grid.voxel_x << ", " << grid.voxel_y << ", " << grid.voxel_z
         << "], \"aggregation\": \"count and fixed-point sums of x/y/z/intensity\"},\n";
  output << "  \"protocol\": {\"warmup_iterations\": " << warmups
         << ", \"measured_iterations\": " << iterations
         << ", \"throughput_basis\": \"input points\", \"cpu_timer\": "
            "\"std::chrono::steady_clock\", \"gpu_kernel_timer\": \"CUDA events around "
            "mapping, stable sort, and reduction\", \"gpu_end_to_end_timer\": "
            "\"std::chrono::steady_clock around allocation, H2D, GPU work, and D2H\"},\n";
  output << "  \"results\": [\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& result = results[i];
    output << "    {\"implementation\": \"" << result.backend << "\", \"timing_scope\": \""
           << result.scope << "\", \"mean_ms\": " << result.statistics.mean
           << ", \"p95_ms\": " << result.statistics.p95
           << ", \"throughput_million_points_per_second\": "
           << result.statistics.million_points_per_second
           << ", \"exact_match_to_cpu\": true, \"mean_speedup_vs_cpu\": ";
    if (result.mean_speedup_vs_cpu > 0.0)
      output << result.mean_speedup_vs_cpu;
    else
      output << "null";
    output << ", \"p95_speedup_vs_cpu\": ";
    if (result.p95_speedup_vs_cpu > 0.0)
      output << result.p95_speedup_vs_cpu;
    else
      output << "null";
    output << "}" << (i + 1 == results.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2)
      throw std::invalid_argument(
          "usage: voxelization_benchmark <nuScenes .pcd.bin sweep> [result.json]");
    const std::string sweep_path = argv[1];
    const std::string output_path =
        argc > 2 ? argv[2] : "benchmark-results/voxelization-m5-local.json";
    constexpr int warmups = 10;
    constexpr int iterations = 100;
    const gve::VoxelGrid grid;
    const auto points = gve::load_nuscenes_lidar_sweep(sweep_path);
    const auto reference = gve::voxelize_cpu(points, grid);
    if (points.empty()) throw std::runtime_error("real nuScenes sweep contains no points");

    gve::VoxelizationResult sink;
    const auto cpu_samples =
        measure_cpu([&] { sink = gve::voxelize_cpu(points, grid); }, warmups, iterations);
    const Statistics cpu = summarize(cpu_samples, points.size());
    std::vector<Result> results = {{"cpu_sort_reduce", "operation", cpu, 0.0, 0.0}};

#ifdef GVE_HAS_CUDA
    const auto cuda_check = gve::voxelize_cuda(points, grid);
    if (!(cuda_check == reference)) throw std::runtime_error("CUDA correctness gate failed");
    for (int i = 0; i < warmups; ++i) sink = gve::voxelize_cuda(points, grid);
    std::vector<double> kernel_samples;
    std::vector<double> end_to_end_samples;
    kernel_samples.reserve(iterations);
    end_to_end_samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
      gve::VoxelCudaTiming timing;
      sink = gve::voxelize_cuda(points, grid, &timing);
      kernel_samples.push_back(timing.kernel_ms);
      end_to_end_samples.push_back(timing.end_to_end_ms);
    }
    const Statistics kernel = summarize(kernel_samples, points.size());
    const Statistics end_to_end = summarize(end_to_end_samples, points.size());
    results.push_back(
        {"cuda_sort_reduce", "kernel_only", kernel, cpu.mean / kernel.mean, cpu.p95 / kernel.p95});
    results.push_back({"cuda_sort_reduce", "end_to_end", end_to_end, cpu.mean / end_to_end.mean,
                       cpu.p95 / end_to_end.p95});
#endif

    write_json(output_path, sweep_path, points, grid, reference, results, warmups, iterations);
    std::cout << "Correctness gate passed. Results written to " << output_path << '\n';
    for (const auto& result : results)
      std::cout << result.backend << " (" << result.scope << "): mean " << result.statistics.mean
                << " ms, p95 " << result.statistics.p95 << " ms\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Benchmark invalid: " << error.what() << '\n';
    return 1;
  }
}
