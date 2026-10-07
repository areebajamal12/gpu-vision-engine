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
#include "gve/gaussian_blur.hpp"
#ifdef GVE_HAS_CUDA
#include <cuda_runtime.h>
#endif

#ifndef GVE_COMPILER_FLAGS
#define GVE_COMPILER_FLAGS "CMake build-type flags; see CMakeCache.txt"
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
                const gve::GaussianParameters& parameters, int warmups, int iterations) {
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
         << "\"\n  },\n";
  output << "  \"workload\": {\"width\": " << width << ", \"height\": " << height
         << ", \"pixel_format\": \"float32_grayscale\", \"kernel_size\": " << parameters.kernel_size
         << ", \"sigma\": " << parameters.sigma << "},\n";
  output
      << "  \"protocol\": {\"warmup_iterations\": " << warmups
      << ", \"measured_iterations\": " << iterations
      << ", \"cpu_timer\": \"std::chrono::steady_clock\", \"gpu_kernel_timer\": \"CUDA events\", "
      << "\"gpu_end_to_end_timer\": \"std::chrono::steady_clock around allocation, H2D, kernels, "
         "D2H\"},\n";
  output << "  \"results\": [\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    output << "    {\"implementation\": \"" << r.backend << "\", \"timing_scope\": \"" << r.scope
           << "\", \"mean_ms\": " << r.statistics.mean << ", \"p95_ms\": " << r.statistics.p95
           << ", \"throughput_megapixels_per_second\": " << r.statistics.megapixels_per_second
           << ", \"max_abs_error\": " << r.max_error << ", \"tolerance\": " << r.tolerance << "}";
    output << (i + 1 == results.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string output_path = argc > 1 ? argv[1] : "benchmark-results/gaussian-blur.json";
    constexpr int width = 1920, height = 1080, warmups = 10, iterations = 100;
    const gve::GaussianParameters parameters{9, 2.0};
    cv::Mat image(height, width, CV_32FC1);
    cv::RNG(0x5EED).fill(image, cv::RNG::UNIFORM, 0.0F, 1.0F);
    const cv::Mat reference = gve::gaussian_blur_opencv(image, parameters);
    std::vector<Result> results;

    const cv::Mat cpu_check = gve::gaussian_blur_cpu(image, parameters);
    const double cpu_error = cv::norm(cpu_check, reference, cv::NORM_INF);
    if (cpu_error > kCpuTolerance) throw std::runtime_error("CPU correctness gate failed");
    cv::Mat sink;
    auto cpu_samples =
        measure_cpu([&] { sink = gve::gaussian_blur_cpu(image, parameters); }, warmups, iterations);
    results.push_back({"cpu_separable", "operation", summarize(cpu_samples, width, height),
                       cpu_error, kCpuTolerance});
    auto opencv_samples = measure_cpu([&] { sink = gve::gaussian_blur_opencv(image, parameters); },
                                      warmups, iterations);
    results.push_back(
        {"opencv_reference", "operation", summarize(opencv_samples, width, height), 0.0, 0.0});

#ifdef GVE_HAS_CUDA
    const cv::Mat cuda_check = gve::gaussian_blur_cuda(image, parameters);
    const double cuda_error = cv::norm(cuda_check, reference, cv::NORM_INF);
    if (cuda_error > kCudaTolerance) throw std::runtime_error("CUDA correctness gate failed");
    for (int i = 0; i < warmups; ++i) sink = gve::gaussian_blur_cuda(image, parameters);
    std::vector<double> kernel_samples, end_to_end_samples;
    for (int i = 0; i < iterations; ++i) {
      gve::CudaTiming timing;
      sink = gve::gaussian_blur_cuda(image, parameters, &timing);
      kernel_samples.push_back(timing.kernel_ms);
      end_to_end_samples.push_back(timing.end_to_end_ms);
    }
    results.push_back({"cuda_naive", "kernel_only", summarize(kernel_samples, width, height),
                       cuda_error, kCudaTolerance});
    results.push_back({"cuda_naive", "end_to_end", summarize(end_to_end_samples, width, height),
                       cuda_error, kCudaTolerance});
#endif

    write_json(output_path, results, width, height, parameters, warmups, iterations);
    std::cout << "Correctness gate passed. Results written to " << output_path << "\n";
    for (const auto& result : results)
      std::cout << result.backend << " (" << result.scope << "): mean " << result.statistics.mean
                << " ms, p95 " << result.statistics.p95 << " ms\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Benchmark invalid: " << error.what() << '\n';
    return 1;
  }
}
