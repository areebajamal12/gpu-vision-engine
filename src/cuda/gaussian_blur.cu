#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>
#include <string>

#include "gve/gaussian_blur.hpp"

namespace gve {
namespace {

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

__global__ void horizontal(const float* input, float* temporary, int width, int height,
                           const float* kernel, int radius) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  float sum = 0.0F;
  for (int k = -radius; k <= radius; ++k) {
    sum += input[y * width + max(0, min(width - 1, x + k))] * kernel[k + radius];
  }
  temporary[y * width + x] = sum;
}

__global__ void vertical(const float* temporary, float* output, int width, int height,
                         const float* kernel, int radius) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  float sum = 0.0F;
  for (int k = -radius; k <= radius; ++k) {
    sum += temporary[max(0, min(height - 1, y + k)) * width + x] * kernel[k + radius];
  }
  output[y * width + x] = sum;
}

void launch(const float* input, float* temporary, float* output, int width, int height,
            const float* kernel, int radius) {
  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  horizontal<<<grid, block>>>(input, temporary, width, height, kernel, radius);
  vertical<<<grid, block>>>(temporary, output, width, height, kernel, radius);
}
}  // namespace

cv::Mat gaussian_blur_cuda(const cv::Mat& input, const GaussianParameters& parameters,
                           CudaTiming* timing) {
  if (input.empty() || input.type() != CV_32FC1 || !input.isContinuous())
    throw std::invalid_argument("input must be a non-empty, continuous CV_32FC1 image");
  const auto host_kernel = make_gaussian_kernel(parameters);
  const size_t image_bytes = input.total() * sizeof(float);
  const size_t kernel_bytes = host_kernel.size() * sizeof(float);
  float *device_input = nullptr, *device_temporary = nullptr, *device_output = nullptr,
        *device_kernel = nullptr;
  cudaEvent_t start, stop;
  const auto total_start = std::chrono::steady_clock::now();
  check(cudaMalloc(&device_input, image_bytes), "cudaMalloc input");
  check(cudaMalloc(&device_temporary, image_bytes), "cudaMalloc temporary");
  check(cudaMalloc(&device_output, image_bytes), "cudaMalloc output");
  check(cudaMalloc(&device_kernel, kernel_bytes), "cudaMalloc kernel");
  check(cudaMemcpy(device_input, input.ptr<float>(), image_bytes, cudaMemcpyHostToDevice),
        "copy input");
  check(cudaMemcpy(device_kernel, host_kernel.data(), kernel_bytes, cudaMemcpyHostToDevice),
        "copy kernel");
  check(cudaEventCreate(&start), "create start event");
  check(cudaEventCreate(&stop), "create stop event");
  check(cudaEventRecord(start), "record start");
  launch(device_input, device_temporary, device_output, input.cols, input.rows, device_kernel,
         parameters.kernel_size / 2);
  check(cudaGetLastError(), "launch kernels");
  check(cudaEventRecord(stop), "record stop");
  check(cudaEventSynchronize(stop), "synchronize kernels");
  float kernel_ms = 0.0F;
  check(cudaEventElapsedTime(&kernel_ms, start, stop), "measure kernels");
  cv::Mat output(input.size(), CV_32FC1);
  check(cudaMemcpy(output.ptr<float>(), device_output, image_bytes, cudaMemcpyDeviceToHost),
        "copy output");
  const auto total_stop = std::chrono::steady_clock::now();
  if (timing) {
    timing->kernel_ms = kernel_ms;
    timing->end_to_end_ms =
        std::chrono::duration<double, std::milli>(total_stop - total_start).count();
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_input);
  cudaFree(device_temporary);
  cudaFree(device_output);
  cudaFree(device_kernel);
  return output;
}

}  // namespace gve
