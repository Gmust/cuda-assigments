#include "filter_utils.h"
#include <plog/Log.h>
#include <algorithm>

namespace cuda_filter
{

    FilterType FilterUtils::stringToFilterType(const std::string &filterName)
    {
        if (filterName == "blur")
            return FilterType::BLUR;
        if (filterName == "sharpen")
            return FilterType::SHARPEN;
        if (filterName == "edge")
            return FilterType::EDGE_DETECTION;
        if (filterName == "emboss")
            return FilterType::EMBOSS;
        if (filterName == "hdr")
            return FilterType::HDR_TONEMAPPING;

        PLOG_WARNING << "Unknown filter type: " << filterName << ". Using default blur filter.";
        return FilterType::BLUR;
    }

    cv::Mat FilterUtils::createFilterKernel(FilterType type, int kernelSize, float intensity)
    {
        // Ensure kernel size is odd
        if (kernelSize % 2 == 0)
        {
            kernelSize++;
        }

        cv::Mat kernel;

        switch (type)
        {
        case FilterType::BLUR:
            kernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F) / (float)(kernelSize * kernelSize);
            break;

        case FilterType::SHARPEN:
            kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
            kernel.at<float>(kernelSize / 2, kernelSize / 2) = 1.0f + 4.0f * intensity;
            if (kernelSize >= 3)
            {
                kernel.at<float>(kernelSize / 2 - 1, kernelSize / 2) = -intensity;
                kernel.at<float>(kernelSize / 2 + 1, kernelSize / 2) = -intensity;
                kernel.at<float>(kernelSize / 2, kernelSize / 2 - 1) = -intensity;
                kernel.at<float>(kernelSize / 2, kernelSize / 2 + 1) = -intensity;
            }
            break;

        case FilterType::EDGE_DETECTION:
            kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
            if (kernelSize >= 3)
            {
                kernel.at<float>(0, 0) = -intensity;
                kernel.at<float>(0, 1) = -intensity;
                kernel.at<float>(0, 2) = -intensity;
                kernel.at<float>(1, 0) = -intensity;
                kernel.at<float>(1, 1) = 8.0f * intensity;
                kernel.at<float>(1, 2) = -intensity;
                kernel.at<float>(2, 0) = -intensity;
                kernel.at<float>(2, 1) = -intensity;
                kernel.at<float>(2, 2) = -intensity;
            }
            break;

        case FilterType::EMBOSS:
            kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
            if (kernelSize >= 3)
            {
                kernel.at<float>(0, 0) = -2.0f * intensity;
                kernel.at<float>(0, 1) = -intensity;
                kernel.at<float>(0, 2) = 0.0f;
                kernel.at<float>(1, 0) = -intensity;
                kernel.at<float>(1, 1) = 1.0f;
                kernel.at<float>(1, 2) = intensity;
                kernel.at<float>(2, 0) = 0.0f;
                kernel.at<float>(2, 1) = intensity;
                kernel.at<float>(2, 2) = 2.0f * intensity;
            }
            break;

        case FilterType::HDR_TONEMAPPING:
            // HDR uses tone mapping kernels, not convolution; return identity
            kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
            kernel.at<float>(kernelSize / 2, kernelSize / 2) = 1.0f;
            break;

        case FilterType::IDENTITY:
        default:
            kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
            kernel.at<float>(kernelSize / 2, kernelSize / 2) = 1.0f;
            break;
        }

        return kernel;
    }

    void FilterUtils::applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        cv::filter2D(input, output, -1, kernel);
    }

} // namespace cuda_filter
