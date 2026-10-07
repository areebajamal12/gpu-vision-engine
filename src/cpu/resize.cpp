#include "gve/resize.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gve {

cv::Mat resize_bilinear_cpu(const cv::Mat& input, cv::Size output_size) {
  if (input.empty() || input.type() != CV_32FC1)
    throw std::invalid_argument("input must be a non-empty CV_32FC1 image");
  if (output_size.width <= 0 || output_size.height <= 0)
    throw std::invalid_argument("output dimensions must be positive");

  cv::Mat output(output_size, CV_32FC1);
  const float scale_x = static_cast<float>(input.cols) / output_size.width;
  const float scale_y = static_cast<float>(input.rows) / output_size.height;
  for (int y = 0; y < output.rows; ++y) {
    const float source_y = (y + 0.5F) * scale_y - 0.5F;
    const int y0_unclamped = static_cast<int>(std::floor(source_y));
    const int y1_unclamped = y0_unclamped + 1;
    const float weight_y = source_y - y0_unclamped;
    const int y0 = std::clamp(y0_unclamped, 0, input.rows - 1);
    const int y1 = std::clamp(y1_unclamped, 0, input.rows - 1);
    const float* row0 = input.ptr<float>(y0);
    const float* row1 = input.ptr<float>(y1);
    float* destination = output.ptr<float>(y);
    for (int x = 0; x < output.cols; ++x) {
      const float source_x = (x + 0.5F) * scale_x - 0.5F;
      const int x0_unclamped = static_cast<int>(std::floor(source_x));
      const int x1_unclamped = x0_unclamped + 1;
      const float weight_x = source_x - x0_unclamped;
      const int x0 = std::clamp(x0_unclamped, 0, input.cols - 1);
      const int x1 = std::clamp(x1_unclamped, 0, input.cols - 1);
      const float top = row0[x0] + weight_x * (row0[x1] - row0[x0]);
      const float bottom = row1[x0] + weight_x * (row1[x1] - row1[x0]);
      destination[x] = top + weight_y * (bottom - top);
    }
  }
  return output;
}

}  // namespace gve
