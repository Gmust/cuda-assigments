#include <gtest/gtest.h>
#include "kernels.h"
#include "filter_utils.h"
#include <opencv2/opencv.hpp>

TEST(ConvolutionTest, IdentityFilter)
{
    // Create a simple test image
    cv::Mat testImage = cv::Mat::ones(100, 100, CV_8UC3) * 128;

    // Create identity filter
    cv::Mat kernel = cuda_filter::FilterUtils::createFilterKernel(
        cuda_filter::FilterType::IDENTITY, 3);

    cv::Mat result;
    cuda_filter::applyFilterGPU(testImage, result, kernel);

    // Check that the result matches the original
    cv::Mat diff;
    cv::absdiff(testImage, result, diff);

    EXPECT_LE(cv::sum(diff)[0], 1e-5);
}

TEST(ConvolutionTest, BlurFilter)
{
    // Create a test image with a single white pixel in the center
    cv::Mat testImage = cv::Mat::zeros(100, 100, CV_8UC1);
    testImage.at<uchar>(50, 50) = 255;

    // Create blur filter
    cv::Mat kernel = cuda_filter::FilterUtils::createFilterKernel(
        cuda_filter::FilterType::BLUR, 3);

    cv::Mat resultGPU, resultCPU;
    cuda_filter::applyFilterGPU(testImage, resultGPU, kernel);
    cuda_filter::FilterUtils::applyFilterCPU(testImage, resultCPU, kernel);

    // Check that GPU result is close to CPU result
    cv::Mat diff;
    cv::absdiff(resultGPU, resultCPU, diff);

    double maxDiff;
    cv::minMaxLoc(diff, nullptr, &maxDiff);

    EXPECT_LE(maxDiff, 1.0);
}
