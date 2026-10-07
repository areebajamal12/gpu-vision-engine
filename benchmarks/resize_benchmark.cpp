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
#include "gve/resize.hpp"

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
constexpr double kCudaNaiveTolerance = 1e-4;
constexpr double kCudaTextureTolerance = 5e-3;

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

Statistics summarize(std::vector<double> values, cv::Size output_size) {
  std::sort(values.begin(), values.end());
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  const size_t p95_index = static_cast<size_t>(std::ceil(values.size() * 0.95)) - 1;
  const double megapixels = static_cast<double>(output_size.width) * output_size.height / 1e6;
  return {mean, values[p95_index], megapixels / (mean / 1000.0)};
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

void write_json(const std::string& path, const std::vector<Result>& results, cv::Size input_size,
                cv::Size output_size, int warmups, int iterations) {
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
  output << "  \"workload\": {\"operation\": \"bilinear_resize\", \"input_width\": "
         << input_size.width << ", \"input_height\": " << input_size.height
         << ", \"output_width\": " << output_size.width
         << ", \"output_height\": " << output_size.height
         << ", \"pixel_format\": \"float32_grayscale\"},\n";
  output
      << "  \"protocol\": {\"warmup_iterations\": " << warmups
      << ", \"measured_iterations\": " << iterations
      << ", \"throughput_basis\": \"output pixels\", \"cpu_timer\": \"std::chrono::steady_clock\", "
      << "\"gpu_kernel_timer\": \"CUDA events\", \"gpu_end_to_end_timer\": "
         "\"std::chrono::steady_clock around allocation, H2D, kernel, D2H\"},\n";
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
    const std::string output_path = argc > 1 ? argv[1] : "benchmark-results/resize-m4-local.json";
    const cv::Size input_size(1920, 1080);
    const cv::Size output_size(1280, 720);
    constexpr int warmups = 10;
    constexpr int iterations = 100;
    cv::Mat image(input_size, CV_32FC1);
    cv::RNG(0xAE512E).fill(image, cv::RNG::UNIFORM, 0.0F, 1.0F);
    const cv::Mat reference = gve::resize_bilinear_opencv(image, output_size);
    std::vector<Result> results;

    const cv::Mat cpu_check = gve::resize_bilinear_cpu(image, output_size);
    const double cpu_error = cv::norm(cpu_check, reference, cv::NORM_INF);
    if (cpu_error > kCpuTolerance) throw std::runtime_error("CPU correctness gate failed");
    cv::Mat sink;
    const auto cpu_samples = measure_cpu(
        [&] { sink = gve::resize_bilinear_cpu(image, output_size); }, warmups, iterations);
    results.push_back({"cpu_bilinear", "operation", summarize(cpu_samples, output_size), cpu_error,
                       kCpuTolerance, 0.0, 0.0});
    const auto opencv_samples = measure_cpu(
        [&] { sink = gve::resize_bilinear_opencv(image, output_size); }, warmups, iterations);
    results.push_back({"opencv_reference", "operation", summarize(opencv_samples, output_size), 0.0,
                       0.0, 0.0, 0.0});

#ifdef GVE_HAS_CUDA
    const cv::Mat naive_check = gve::resize_bilinear_cuda(image, output_size);
    const double naive_error = cv::norm(naive_check, reference, cv::NORM_INF);
    if (naive_error > kCudaNaiveTolerance)
      throw std::runtime_error("CUDA naive correctness gate failed");
    for (int i = 0; i < warmups; ++i) sink = gve::resize_bilinear_cuda(image, output_size);
    std::vector<double> naive_kernel_samples;
    std::vector<double> naive_end_to_end_samples;
    for (int i = 0; i < iterations; ++i) {
      gve::CudaTiming timing;
      sink = gve::resize_bilinear_cuda(image, output_size, &timing);
      naive_kernel_samples.push_back(timing.kernel_ms);
      naive_end_to_end_samples.push_back(timing.end_to_end_ms);
    }
    const Statistics naive_kernel = summarize(naive_kernel_samples, output_size);
    const Statistics naive_end_to_end = summarize(naive_end_to_end_samples, output_size);
    results.push_back(
        {"cuda_naive", "kernel_only", naive_kernel, naive_error, kCudaNaiveTolerance, 1.0, 1.0});
    results.push_back(
        {"cuda_naive", "end_to_end", naive_end_to_end, naive_error, kCudaNaiveTolerance, 1.0, 1.0});

    const cv::Mat texture_check = gve::resize_bilinear_cuda_texture(image, output_size);
    const double texture_error = cv::norm(texture_check, reference, cv::NORM_INF);
    if (texture_error > kCudaTextureTolerance)
      throw std::runtime_error("CUDA texture correctness gate failed");
    for (int i = 0; i < warmups; ++i) sink = gve::resize_bilinear_cuda_texture(image, output_size);
    std::vector<double> texture_kernel_samples;
    std::vector<double> texture_end_to_end_samples;
    for (int i = 0; i < iterations; ++i) {
      gve::CudaTiming timing;
      sink = gve::resize_bilinear_cuda_texture(image, output_size, &timing);
      texture_kernel_samples.push_back(timing.kernel_ms);
      texture_end_to_end_samples.push_back(timing.end_to_end_ms);
    }
    const Statistics texture_kernel = summarize(texture_kernel_samples, output_size);
    const Statistics texture_end_to_end = summarize(texture_end_to_end_samples, output_size);
    results.push_back({"cuda_texture", "kernel_only", texture_kernel, texture_error,
                       kCudaTextureTolerance, naive_kernel.mean / texture_kernel.mean,
                       naive_kernel.p95 / texture_kernel.p95});
    results.push_back({"cuda_texture", "end_to_end", texture_end_to_end, texture_error,
                       kCudaTextureTolerance, naive_end_to_end.mean / texture_end_to_end.mean,
                       naive_end_to_end.p95 / texture_end_to_end.p95});
#endif

    write_json(output_path, results, input_size, output_size, warmups, iterations);
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
