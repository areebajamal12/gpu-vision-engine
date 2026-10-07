#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "gve/sobel.hpp"

namespace gve {

cv::Mat sobel_opencv(const cv::Mat& input) {
  if (input.empty() || input.type() != CV_32FC1)
    throw std::invalid_argument("input must be a non-empty CV_32FC1 image");
  cv::Mat gradient_x;
  cv::Mat gradient_y;
  cv::Mat output;
  cv::Sobel(input, gradient_x, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
  cv::Sobel(input, gradient_y, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
  cv::magnitude(gradient_x, gradient_y, output);
  return output;
}

}  // namespace gve
