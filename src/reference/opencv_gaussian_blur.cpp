#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "gve/gaussian_blur.hpp"

namespace gve {

cv::Mat gaussian_blur_opencv(const cv::Mat& input, const GaussianParameters& parameters) {
  if (input.empty() || input.type() != CV_32FC1) {
    throw std::invalid_argument("input must be a non-empty CV_32FC1 image");
  }
  cv::Mat output;
  cv::GaussianBlur(input, output, cv::Size(parameters.kernel_size, parameters.kernel_size),
                   parameters.sigma, parameters.sigma, cv::BORDER_REPLICATE);
  return output;
}

}  // namespace gve
