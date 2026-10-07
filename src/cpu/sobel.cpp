#include "gve/sobel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gve {

cv::Mat sobel_cpu(const cv::Mat& input) {
  if (input.empty() || input.type() != CV_32FC1)
    throw std::invalid_argument("input must be a non-empty CV_32FC1 image");

  constexpr int kHorizontal[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  constexpr int kVertical[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
  cv::Mat output(input.size(), CV_32FC1);
  for (int y = 0; y < input.rows; ++y) {
    float* destination = output.ptr<float>(y);
    for (int x = 0; x < input.cols; ++x) {
      float gradient_x = 0.0F;
      float gradient_y = 0.0F;
      for (int row = -1; row <= 1; ++row) {
        const float* source = input.ptr<float>(std::clamp(y + row, 0, input.rows - 1));
        for (int column = -1; column <= 1; ++column) {
          const float value = source[std::clamp(x + column, 0, input.cols - 1)];
          gradient_x += value * kHorizontal[row + 1][column + 1];
          gradient_y += value * kVertical[row + 1][column + 1];
        }
      }
      destination[x] = std::sqrt(gradient_x * gradient_x + gradient_y * gradient_y);
    }
  }
  return output;
}

}  // namespace gve
