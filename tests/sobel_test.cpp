#include "gve/sobel.hpp"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

namespace {
constexpr double kCpuTolerance = 1e-4;
constexpr double kCudaTolerance = 1e-3;

cv::Mat sobel_test_image(int width = 257, int height = 193) {
  cv::Mat image(height, width, CV_32FC1);
  cv::RNG(0x50BE1).fill(image, cv::RNG::UNIFORM, 0.0F, 1.0F);
  return image;
}

double sobel_max_error(const cv::Mat& actual, const cv::Mat& expected) {
  return cv::norm(actual, expected, cv::NORM_INF);
}
}  // namespace

TEST(SobelCpu, MatchesOpenCvReference) {
  const auto image = sobel_test_image();
  EXPECT_LE(sobel_max_error(gve::sobel_cpu(image), gve::sobel_opencv(image)), kCpuTolerance);
}

TEST(SobelCpu, HandlesMilestoneResolution) {
  const auto output = gve::sobel_cpu(sobel_test_image(1920, 1080));
  EXPECT_EQ(output.cols, 1920);
  EXPECT_EQ(output.rows, 1080);
}

TEST(SobelCpu, RejectsInvalidInput) {
  EXPECT_THROW(gve::sobel_cpu(cv::Mat(8, 8, CV_8UC1)), std::invalid_argument);
}

#ifdef GVE_HAS_CUDA
TEST(SobelCuda, MatchesOpenCvReference) {
  const auto image = sobel_test_image();
  EXPECT_LE(sobel_max_error(gve::sobel_cuda(image), gve::sobel_opencv(image)), kCudaTolerance);
}

TEST(SobelCudaTiled, MatchesOpenCvReference) {
  const auto image = sobel_test_image();
  EXPECT_LE(sobel_max_error(gve::sobel_cuda_tiled(image), gve::sobel_opencv(image)),
            kCudaTolerance);
}
#endif
