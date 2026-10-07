#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>
#include <string>

#include "gve/sobel.hpp"

namespace gve {
namespace {

void check_sobel(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

__device__ float magnitude_at(const float values[3][3]) {
  const float gradient_x = -values[0][0] + values[0][2] - 2.0F * values[1][0] +
                           2.0F * values[1][2] - values[2][0] + values[2][2];
  const float gradient_y = -values[0][0] - 2.0F * values[0][1] - values[0][2] + values[2][0] +
                           2.0F * values[2][1] + values[2][2];
  return sqrtf(gradient_x * gradient_x + gradient_y * gradient_y);
}

__global__ void sobel_naive_kernel(const float* input, float* output, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  float values[3][3];
  for (int row = -1; row <= 1; ++row) {
    const int source_y = max(0, min(height - 1, y + row));
    for (int column = -1; column <= 1; ++column) {
      const int source_x = max(0, min(width - 1, x + column));
      values[row + 1][column + 1] = input[source_y * width + source_x];
    }
  }
  output[y * width + x] = magnitude_at(values);
}

__global__ void sobel_tiled_kernel(const float* input, float* output, int width, int height) {
  extern __shared__ float tile[];
  const int tile_width = blockDim.x + 2;
  const int tile_height = blockDim.y + 2;
  const int tile_elements = tile_width * tile_height;
  const int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
  const int thread_count = blockDim.x * blockDim.y;
  const int block_x = blockIdx.x * blockDim.x;
  const int block_y = blockIdx.y * blockDim.y;

  for (int index = thread_index; index < tile_elements; index += thread_count) {
    const int local_y = index / tile_width;
    const int local_x = index % tile_width;
    const int source_x = max(0, min(width - 1, block_x + local_x - 1));
    const int source_y = max(0, min(height - 1, block_y + local_y - 1));
    tile[index] = input[source_y * width + source_x];
  }
  __syncthreads();

  const int x = block_x + threadIdx.x;
  const int y = block_y + threadIdx.y;
  if (x >= width || y >= height) return;
  float values[3][3];
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      values[row][column] = tile[(threadIdx.y + row) * tile_width + threadIdx.x + column];
  output[y * width + x] = magnitude_at(values);
}

void launch_sobel_naive(const float* input, float* output, int width, int height) {
  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  sobel_naive_kernel<<<grid, block>>>(input, output, width, height);
}

void launch_sobel_tiled(const float* input, float* output, int width, int height) {
  const dim3 block(32, 8);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  const size_t shared_bytes = (block.x + 2) * (block.y + 2) * sizeof(float);
  sobel_tiled_kernel<<<grid, block, shared_bytes>>>(input, output, width, height);
}

using SobelLaunch = void (*)(const float*, float*, int, int);

cv::Mat run_sobel_cuda(const cv::Mat& input, CudaTiming* timing, SobelLaunch launch) {
  if (input.empty() || input.type() != CV_32FC1 || !input.isContinuous())
    throw std::invalid_argument("input must be a non-empty, continuous CV_32FC1 image");
  const size_t image_bytes = input.total() * sizeof(float);
  float* device_input = nullptr;
  float* device_output = nullptr;
  cudaEvent_t start;
  cudaEvent_t stop;
  const auto total_start = std::chrono::steady_clock::now();
  check_sobel(cudaMalloc(&device_input, image_bytes), "cudaMalloc input");
  check_sobel(cudaMalloc(&device_output, image_bytes), "cudaMalloc output");
  check_sobel(cudaMemcpy(device_input, input.ptr<float>(), image_bytes, cudaMemcpyHostToDevice),
              "copy input");
  check_sobel(cudaEventCreate(&start), "create start event");
  check_sobel(cudaEventCreate(&stop), "create stop event");
  check_sobel(cudaEventRecord(start), "record start");
  launch(device_input, device_output, input.cols, input.rows);
  check_sobel(cudaGetLastError(), "launch Sobel kernel");
  check_sobel(cudaEventRecord(stop), "record stop");
  check_sobel(cudaEventSynchronize(stop), "synchronize Sobel kernel");
  float kernel_ms = 0.0F;
  check_sobel(cudaEventElapsedTime(&kernel_ms, start, stop), "measure Sobel kernel");
  cv::Mat output(input.size(), CV_32FC1);
  check_sobel(cudaMemcpy(output.ptr<float>(), device_output, image_bytes, cudaMemcpyDeviceToHost),
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
  cudaFree(device_output);
  return output;
}

}  // namespace

cv::Mat sobel_cuda(const cv::Mat& input, CudaTiming* timing) {
  return run_sobel_cuda(input, timing, launch_sobel_naive);
}

cv::Mat sobel_cuda_tiled(const cv::Mat& input, CudaTiming* timing) {
  return run_sobel_cuda(input, timing, launch_sobel_tiled);
}

}  // namespace gve
