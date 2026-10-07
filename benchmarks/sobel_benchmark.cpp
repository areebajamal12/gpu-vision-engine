#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "environment.hpp"
#include "gve/sobel.hpp"

#ifndef GVE_COMPILER_FLAGS
#define GVE_COMPILER_FLAGS "CMake build-type flags; see CMakeCache.txt"
#endif
#ifndef GVE_CUDA_COMPILER
#define GVE_CUDA_COMPILER "not enabled"
#define GVE_CUDA_FLAGS "not enabled"
#endif

namespace {
using Clock = std::chrono::steady_clock;
constexpr double kCpuTolerance = 1e-4;
constexpr double kCudaTolerance = 1e-3;

struct Statistics {
  double mean;
  double p95;
  double megapixels_per_second;
};

struct Result {
  std::string backend;
  std::string scope;
  Statistics statistics;
  double max_error;
  double tolerance;
  double mean_speedup_vs_naive;
  double p95_speedup_vs_naive;
};

Statistics summarize(std::vector<double> values, int width, int height) {
  std::sort(values.begin(), values.end());
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  const size_t p95_index = static_cast<size_t>(std::ceil(values.size() * 0.95)) - 1;
  return {mean, values[p95_index], (static_cast<double>(width) * height / 1e6) / (mean / 1000.0)};
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

void write_json(const std::string& path, const std::vector<Result>& results, int width, int height,
                int warmups, int iterations) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot open output file: " + path);
  output << std::fixed << std::setprecision(6);
  output << "{\n  \"schema_version\": 1,\n  \"correctness_gate_passed\": true,\n";
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
  output << "  \"workload\": {\"operation\": \"sobel_magnitude_3x3\", \"width\": " << width
         << ", \"height\": " << height << ", \"pixel_format\": \"float32_grayscale\", "
         << "\"border_mode\": \"replicate\"},\n";
  output
      << "  \"protocol\": {\"warmup_iterations\": " << warmups
      << ", \"measured_iterations\": " << iterations
      << ", \"cpu_timer\": \"std::chrono::steady_clock\", \"gpu_kernel_timer\": \"CUDA events\", "
      << "\"gpu_end_to_end_timer\": \"std::chrono::steady_clock around allocation, H2D, kernel, "
         "D2H\"},\n";
  output << "  \"results\": [\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto& result = results[i];
    output << "    {\"implementation\": \"" << result.backend << "\", \"timing_scope\": \""
           << result.scope << "\", \"mean_ms\": " << result.statistics.mean
           << ", \"p95_ms\": " << result.statistics.p95
           << ", \"throughput_megapixels_per_second\": " << result.statistics.megapixels_per_second
           << ", \"max_abs_error\": " << result.max_error << ", \"tolerance\": " << result.tolerance
           << ", \"mean_speedup_vs_naive\": ";
    if (result.mean_speedup_vs_naive > 0.0)
      output << result.mean_speedup_vs_naive;
    else
      output << "null";
    output << ", \"p95_speedup_vs_naive\": ";
    if (result.p95_speedup_vs_naive > 0.0)
      output << result.p95_speedup_vs_naive;
    else
      output << "null";
    output << "}" << (i + 1 == results.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string output_path = argc > 1 ? argv[1] : "benchmark-results/sobel-m3-local.json";
    constexpr int width = 1920;
    constexpr int height = 1080;
    constexpr int warmups = 10;
    constexpr int iterations = 100;
    cv::Mat image(height, width, CV_32FC1);
    cv::RNG(0x50BE1).fill(image, cv::RNG::UNIFORM, 0.0F, 1.0F);
    const cv::Mat reference = gve::sobel_opencv(image);
    std::vector<Result> results;

    const cv::Mat cpu_check = gve::sobel_cpu(image);
    const double cpu_error = cv::norm(cpu_check, reference, cv::NORM_INF);
    if (cpu_error > kCpuTolerance) throw std::runtime_error("CPU correctness gate failed");
    cv::Mat sink;
    const auto cpu_samples =
        measure_cpu([&] { sink = gve::sobel_cpu(image); }, warmups, iterations);
    results.push_back({"cpu_sobel", "operation", summarize(cpu_samples, width, height), cpu_error,
                       kCpuTolerance, 0.0, 0.0});
    const auto opencv_samples =
        measure_cpu([&] { sink = gve::sobel_opencv(image); }, warmups, iterations);
    results.push_back({"opencv_reference", "operation", summarize(opencv_samples, width, height),
                       0.0, 0.0, 0.0, 0.0});

#ifdef GVE_HAS_CUDA
    const cv::Mat naive_check = gve::sobel_cuda(image);
    const double naive_error = cv::norm(naive_check, reference, cv::NORM_INF);
    if (naive_error > kCudaTolerance)
      throw std::runtime_error("CUDA naive correctness gate failed");
    for (int i = 0; i < warmups; ++i) sink = gve::sobel_cuda(image);
    std::vector<double> naive_kernel_samples;
    std::vector<double> naive_end_to_end_samples;
    for (int i = 0; i < iterations; ++i) {
      gve::CudaTiming timing;
      sink = gve::sobel_cuda(image, &timing);
      naive_kernel_samples.push_back(timing.kernel_ms);
      naive_end_to_end_samples.push_back(timing.end_to_end_ms);
    }
    const Statistics naive_kernel = summarize(naive_kernel_samples, width, height);
    const Statistics naive_end_to_end = summarize(naive_end_to_end_samples, width, height);
    results.push_back(
        {"cuda_naive", "kernel_only", naive_kernel, naive_error, kCudaTolerance, 1.0, 1.0});
    results.push_back(
        {"cuda_naive", "end_to_end", naive_end_to_end, naive_error, kCudaTolerance, 1.0, 1.0});

    const cv::Mat tiled_check = gve::sobel_cuda_tiled(image);
    const double tiled_error = cv::norm(tiled_check, reference, cv::NORM_INF);
    if (tiled_error > kCudaTolerance)
      throw std::runtime_error("CUDA tiled correctness gate failed");
    for (int i = 0; i < warmups; ++i) sink = gve::sobel_cuda_tiled(image);
    std::vector<double> tiled_kernel_samples;
    std::vector<double> tiled_end_to_end_samples;
    for (int i = 0; i < iterations; ++i) {
      gve::CudaTiming timing;
      sink = gve::sobel_cuda_tiled(image, &timing);
      tiled_kernel_samples.push_back(timing.kernel_ms);
      tiled_end_to_end_samples.push_back(timing.end_to_end_ms);
    }
    const Statistics tiled_kernel = summarize(tiled_kernel_samples, width, height);
    const Statistics tiled_end_to_end = summarize(tiled_end_to_end_samples, width, height);
    results.push_back({"cuda_tiled", "kernel_only", tiled_kernel, tiled_error, kCudaTolerance,
                       naive_kernel.mean / tiled_kernel.mean, naive_kernel.p95 / tiled_kernel.p95});
    results.push_back({"cuda_tiled", "end_to_end", tiled_end_to_end, tiled_error, kCudaTolerance,
                       naive_end_to_end.mean / tiled_end_to_end.mean,
                       naive_end_to_end.p95 / tiled_end_to_end.p95});
#endif

    write_json(output_path, results, width, height, warmups, iterations);
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
