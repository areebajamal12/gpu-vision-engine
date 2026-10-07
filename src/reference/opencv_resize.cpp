#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "gve/resize.hpp"

namespace gve {

cv::Mat resize_bilinear_opencv(const cv::Mat& input, cv::Size output_size) {
  if (input.empty() || input.type() != CV_32FC1)
    throw std::invalid_argument("input must be a non-empty CV_32FC1 image");
  if (output_size.width <= 0 || output_size.height <= 0)
    throw std::invalid_argument("output dimensions must be positive");
  cv::Mat output;
  cv::resize(input, output, output_size, 0.0, 0.0, cv::INTER_LINEAR);
  return output;
}

}  // namespace gve
