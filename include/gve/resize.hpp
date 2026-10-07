#pragma once

#include <opencv2/core.hpp>

#include "gve/gaussian_blur.hpp"

namespace gve {

cv::Mat resize_bilinear_cpu(const cv::Mat& input, cv::Size output_size);
cv::Mat resize_bilinear_opencv(const cv::Mat& input, cv::Size output_size);

#ifdef GVE_HAS_CUDA
cv::Mat resize_bilinear_cuda(const cv::Mat& input, cv::Size output_size,
                             CudaTiming* timing = nullptr);
cv::Mat resize_bilinear_cuda_texture(const cv::Mat& input, cv::Size output_size,
                                     CudaTiming* timing = nullptr);
#endif

}  // namespace gve
