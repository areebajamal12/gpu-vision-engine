#pragma once

#include <opencv2/core.hpp>

#include "gve/gaussian_blur.hpp"

namespace gve {

cv::Mat sobel_cpu(const cv::Mat& input);
cv::Mat sobel_opencv(const cv::Mat& input);

#ifdef GVE_HAS_CUDA
cv::Mat sobel_cuda(const cv::Mat& input, CudaTiming* timing = nullptr);
cv::Mat sobel_cuda_tiled(const cv::Mat& input, CudaTiming* timing = nullptr);
#endif

}  // namespace gve
