#include "gve/resize.hpp"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

namespace {
constexpr double kCpuTolerance = 1e-4;
constexpr double kCudaNaiveTolerance = 1e-4;
constexpr double kCudaTextureTolerance = 5e-3;

cv::Mat resize_test_image(int width = 257, int height = 193) {
  cv::Mat image(height, width, CV_32FC1);
  cv::RNG(0xAE512E).fill(image, cv::RNG::UNIFORM, 0.0F, 1.0F);
  return image;
}

double resize_max_error(const cv::Mat& actual, const cv::Mat& expected) {
  return cv::norm(actual, expected, cv::NORM_INF);
}
}  // namespace

TEST(ResizeCpu, MatchesOpenCvReference) {
  const auto image = resize_test_image();
  const cv::Size output_size(171, 127);
  EXPECT_LE(resize_max_error(gve::resize_bilinear_cpu(image, output_size),
                             gve::resize_bilinear_opencv(image, output_size)),
            kCpuTolerance);
}

TEST(ResizeCpu, HandlesMilestoneDimensions) {
  const auto output = gve::resize_bilinear_cpu(resize_test_image(1920, 1080), {1280, 720});
  EXPECT_EQ(output.cols, 1280);
  EXPECT_EQ(output.rows, 720);
}

TEST(ResizeCpu, RejectsInvalidDimensions) {
  EXPECT_THROW(gve::resize_bilinear_cpu(resize_test_image(), {0, 720}), std::invalid_argument);
}

#ifdef GVE_HAS_CUDA
TEST(ResizeCuda, MatchesOpenCvReference) {
  const auto image = resize_test_image();
  const cv::Size output_size(171, 127);
  EXPECT_LE(resize_max_error(gve::resize_bilinear_cuda(image, output_size),
                             gve::resize_bilinear_opencv(image, output_size)),
            kCudaNaiveTolerance);
}

TEST(ResizeCudaTexture, MatchesOpenCvReference) {
  const auto image = resize_test_image();
  const cv::Size output_size(171, 127);
  EXPECT_LE(resize_max_error(gve::resize_bilinear_cuda_texture(image, output_size),
                             gve::resize_bilinear_opencv(image, output_size)),
            kCudaTextureTolerance);
}
#endif
