#include <gtest/gtest.h>
#include "filter_utils.h"

TEST(FilterUtilsTest, StringToFilterType)
{
    EXPECT_EQ(cuda_filter::FilterUtils::stringToFilterType("blur"), cuda_filter::FilterType::BLUR);
    EXPECT_EQ(cuda_filter::FilterUtils::stringToFilterType("sharpen"), cuda_filter::FilterType::SHARPEN);
    EXPECT_EQ(cuda_filter::FilterUtils::stringToFilterType("edge"), cuda_filter::FilterType::EDGE_DETECTION);
    EXPECT_EQ(cuda_filter::FilterUtils::stringToFilterType("emboss"), cuda_filter::FilterType::EMBOSS);

    // Unknown filter should default to BLUR
    EXPECT_EQ(cuda_filter::FilterUtils::stringToFilterType("unknown"), cuda_filter::FilterType::BLUR);
}

TEST(FilterUtilsTest, CreateFilterKernel)
{
    // Test identity kernel
    cv::Mat identityKernel = cuda_filter::FilterUtils::createFilterKernel(
        cuda_filter::FilterType::IDENTITY, 3);

    EXPECT_EQ(identityKernel.rows, 3);
    EXPECT_EQ(identityKernel.cols, 3);
    EXPECT_FLOAT_EQ(identityKernel.at<float>(1, 1), 1.0f);
    EXPECT_FLOAT_EQ(identityKernel.at<float>(0, 0), 0.0f);

    // Test blur kernel
    cv::Mat blurKernel = cuda_filter::FilterUtils::createFilterKernel(
        cuda_filter::FilterType::BLUR, 3);

    EXPECT_EQ(blurKernel.rows, 3);
    EXPECT_EQ(blurKernel.cols, 3);
    EXPECT_FLOAT_EQ(blurKernel.at<float>(0, 0), 1.0f / 9.0f);
    EXPECT_FLOAT_EQ(blurKernel.at<float>(1, 1), 1.0f / 9.0f);

    // Kernel size should be forced to odd
    cv::Mat oddKernel = cuda_filter::FilterUtils::createFilterKernel(
        cuda_filter::FilterType::BLUR, 4); // Request size 4

    EXPECT_EQ(oddKernel.rows, 5); // Should be increased to 5
}
