#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace gve {

struct GaussianParameters {
  int kernel_size = 9;
  double sigma = 2.0;
};

std::vector<float> make_gaussian_kernel(const GaussianParameters& parameters);
cv::Mat gaussian_blur_cpu(const cv::Mat& input, const GaussianParameters& parameters);
cv::Mat gaussian_blur_opencv(const cv::Mat& input, const GaussianParameters& parameters);

struct CudaTiming {
  double kernel_ms = 0.0;
  double end_to_end_ms = 0.0;
};

#ifdef GVE_HAS_CUDA
cv::Mat gaussian_blur_cuda(const cv::Mat& input, const GaussianParameters& parameters,
                           CudaTiming* timing = nullptr);
cv::Mat gaussian_blur_cuda_tiled(const cv::Mat& input, const GaussianParameters& parameters,
                                 CudaTiming* timing = nullptr);
#endif

}  // namespace gve
