#include "gve/gaussian_blur.hpp"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

namespace {
constexpr double kCpuTolerance = 1e-4;
constexpr double kCudaTolerance = 1e-3;

cv::Mat test_image(int width = 257, int height = 193) {
  cv::Mat image(height, width, CV_32FC1);
  cv::RNG rng(0x5EED);
  rng.fill(image, cv::RNG::UNIFORM, 0.0F, 1.0F);
  return image;
}

double max_error(const cv::Mat& actual, const cv::Mat& expected) {
  return cv::norm(actual, expected, cv::NORM_INF);
}
}  // namespace

TEST(GaussianBlurCpu, MatchesOpenCvReference) {
  const gve::GaussianParameters parameters{9, 2.0};
  const auto image = test_image();
  EXPECT_LE(max_error(gve::gaussian_blur_cpu(image, parameters),
                      gve::gaussian_blur_opencv(image, parameters)),
            kCpuTolerance);
}

TEST(GaussianBlurCpu, HandlesMilestoneResolution) {
  const gve::GaussianParameters parameters{9, 2.0};
  const auto output = gve::gaussian_blur_cpu(test_image(1920, 1080), parameters);
  EXPECT_EQ(output.cols, 1920);
  EXPECT_EQ(output.rows, 1080);
}

TEST(GaussianBlurCpu, RejectsInvalidParameters) {
  EXPECT_THROW(gve::gaussian_blur_cpu(test_image(), {4, 2.0}), std::invalid_argument);
}

#ifdef GVE_HAS_CUDA
TEST(GaussianBlurCuda, MatchesOpenCvReference) {
  const gve::GaussianParameters parameters{9, 2.0};
  const auto image = test_image();
  EXPECT_LE(max_error(gve::gaussian_blur_cuda(image, parameters),
                      gve::gaussian_blur_opencv(image, parameters)),
            kCudaTolerance);
}
#endif
