#include "gve/gaussian_blur.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace gve {

namespace {
void validate(const cv::Mat& input, const GaussianParameters& parameters) {
  if (input.empty() || input.type() != CV_32FC1) {
    throw std::invalid_argument("input must be a non-empty CV_32FC1 image");
  }
  if (parameters.kernel_size <= 0 || parameters.kernel_size % 2 == 0 || parameters.sigma <= 0.0) {
    throw std::invalid_argument("kernel size must be positive and odd, and sigma must be positive");
  }
}
}  // namespace

std::vector<float> make_gaussian_kernel(const GaussianParameters& parameters) {
  if (parameters.kernel_size <= 0 || parameters.kernel_size % 2 == 0 || parameters.sigma <= 0.0) {
    throw std::invalid_argument("invalid Gaussian parameters");
  }
  const cv::Mat kernel = cv::getGaussianKernel(parameters.kernel_size, parameters.sigma, CV_32F);
  return {kernel.ptr<float>(), kernel.ptr<float>() + kernel.rows};
}

cv::Mat gaussian_blur_cpu(const cv::Mat& input, const GaussianParameters& parameters) {
  validate(input, parameters);
  const auto kernel = make_gaussian_kernel(parameters);
  const int radius = parameters.kernel_size / 2;
  cv::Mat temporary(input.size(), CV_32FC1);
  cv::Mat output(input.size(), CV_32FC1);

  for (int y = 0; y < input.rows; ++y) {
    const float* source = input.ptr<float>(y);
    float* destination = temporary.ptr<float>(y);
    for (int x = 0; x < input.cols; ++x) {
      float sum = 0.0F;
      for (int k = -radius; k <= radius; ++k) {
        sum += source[std::clamp(x + k, 0, input.cols - 1)] * kernel[k + radius];
      }
      destination[x] = sum;
    }
  }

  for (int y = 0; y < input.rows; ++y) {
    float* destination = output.ptr<float>(y);
    for (int x = 0; x < input.cols; ++x) {
      float sum = 0.0F;
      for (int k = -radius; k <= radius; ++k) {
        sum += temporary.ptr<float>(std::clamp(y + k, 0, input.rows - 1))[x] * kernel[k + radius];
      }
      destination[x] = sum;
    }
  }
  return output;
}

}  // namespace gve
