#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include "gve/resize.hpp"

namespace gve {
namespace {

void check_resize(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

__global__ void resize_naive_kernel(const float* input, float* output, int input_width,
                                    int input_height, int output_width, int output_height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= output_width || y >= output_height) return;
  const float scale_x = static_cast<float>(input_width) / output_width;
  const float scale_y = static_cast<float>(input_height) / output_height;
  const float source_x = (x + 0.5F) * scale_x - 0.5F;
  const float source_y = (y + 0.5F) * scale_y - 0.5F;
  const int x0_unclamped = static_cast<int>(floorf(source_x));
  const int y0_unclamped = static_cast<int>(floorf(source_y));
  const int x1_unclamped = x0_unclamped + 1;
  const int y1_unclamped = y0_unclamped + 1;
  const float weight_x = source_x - x0_unclamped;
  const float weight_y = source_y - y0_unclamped;
  const int x0 = max(0, min(input_width - 1, x0_unclamped));
  const int x1 = max(0, min(input_width - 1, x1_unclamped));
  const int y0 = max(0, min(input_height - 1, y0_unclamped));
  const int y1 = max(0, min(input_height - 1, y1_unclamped));
  const float top = input[y0 * input_width + x0] +
                    weight_x * (input[y0 * input_width + x1] - input[y0 * input_width + x0]);
  const float bottom = input[y1 * input_width + x0] +
                       weight_x * (input[y1 * input_width + x1] - input[y1 * input_width + x0]);
  output[y * output_width + x] = top + weight_y * (bottom - top);
}

__global__ void resize_texture_kernel(cudaTextureObject_t texture, float* output, int input_width,
                                      int input_height, int output_width, int output_height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= output_width || y >= output_height) return;
  const float source_x = (x + 0.5F) * static_cast<float>(input_width) / output_width - 0.5F;
  const float source_y = (y + 0.5F) * static_cast<float>(input_height) / output_height - 0.5F;
  output[y * output_width + x] = tex2D<float>(texture, source_x + 0.5F, source_y + 0.5F);
}

enum class ResizeBackend { kNaive, kTexture };

cv::Mat run_resize_cuda(const cv::Mat& input, cv::Size output_size, CudaTiming* timing,
                        ResizeBackend backend) {
  if (input.empty() || input.type() != CV_32FC1 || !input.isContinuous())
    throw std::invalid_argument("input must be a non-empty, continuous CV_32FC1 image");
  if (output_size.width <= 0 || output_size.height <= 0)
    throw std::invalid_argument("output dimensions must be positive");
  const size_t input_bytes = input.total() * sizeof(float);
  const size_t output_bytes =
      static_cast<size_t>(output_size.width) * output_size.height * sizeof(float);
  float* device_input = nullptr;
  float* device_output = nullptr;
  cudaTextureObject_t texture = 0;
  cudaEvent_t start;
  cudaEvent_t stop;
  const auto total_start = std::chrono::steady_clock::now();
  check_resize(cudaMalloc(&device_input, input_bytes), "cudaMalloc input");
  check_resize(cudaMalloc(&device_output, output_bytes), "cudaMalloc output");
  check_resize(cudaMemcpy(device_input, input.ptr<float>(), input_bytes, cudaMemcpyHostToDevice),
               "copy input");
  if (backend == ResizeBackend::kTexture) {
    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypePitch2D;
    resource.res.pitch2D.devPtr = device_input;
    resource.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    resource.res.pitch2D.width = input.cols;
    resource.res.pitch2D.height = input.rows;
    resource.res.pitch2D.pitchInBytes = input.cols * sizeof(float);
    cudaTextureDesc description{};
    description.addressMode[0] = cudaAddressModeClamp;
    description.addressMode[1] = cudaAddressModeClamp;
    description.filterMode = cudaFilterModeLinear;
    description.readMode = cudaReadModeElementType;
    description.normalizedCoords = 0;
    check_resize(cudaCreateTextureObject(&texture, &resource, &description, nullptr),
                 "create texture object");
  }
  check_resize(cudaEventCreate(&start), "create start event");
  check_resize(cudaEventCreate(&stop), "create stop event");
  const dim3 block(32, 8);
  const dim3 grid((output_size.width + block.x - 1) / block.x,
                  (output_size.height + block.y - 1) / block.y);
  check_resize(cudaEventRecord(start), "record start");
  if (backend == ResizeBackend::kNaive)
    resize_naive_kernel<<<grid, block>>>(device_input, device_output, input.cols, input.rows,
                                         output_size.width, output_size.height);
  else
    resize_texture_kernel<<<grid, block>>>(texture, device_output, input.cols, input.rows,
                                           output_size.width, output_size.height);
  check_resize(cudaGetLastError(), "launch resize kernel");
  check_resize(cudaEventRecord(stop), "record stop");
  check_resize(cudaEventSynchronize(stop), "synchronize resize kernel");
  float kernel_ms = 0.0F;
  check_resize(cudaEventElapsedTime(&kernel_ms, start, stop), "measure resize kernel");
  cv::Mat output(output_size, CV_32FC1);
  check_resize(cudaMemcpy(output.ptr<float>(), device_output, output_bytes, cudaMemcpyDeviceToHost),
               "copy output");
  const auto total_stop = std::chrono::steady_clock::now();
  if (timing) {
    timing->kernel_ms = kernel_ms;
    timing->end_to_end_ms =
        std::chrono::duration<double, std::milli>(total_stop - total_start).count();
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (texture) cudaDestroyTextureObject(texture);
  cudaFree(device_input);
  cudaFree(device_output);
  return output;
}

}  // namespace

cv::Mat resize_bilinear_cuda(const cv::Mat& input, cv::Size output_size, CudaTiming* timing) {
  return run_resize_cuda(input, output_size, timing, ResizeBackend::kNaive);
}

cv::Mat resize_bilinear_cuda_texture(const cv::Mat& input, cv::Size output_size,
                                     CudaTiming* timing) {
  return run_resize_cuda(input, output_size, timing, ResizeBackend::kTexture);
}

}  // namespace gve
