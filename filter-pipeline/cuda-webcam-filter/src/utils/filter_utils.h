#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace cuda_filter
{

    enum class FilterType
    {
        BLUR,
        SHARPEN,
        EDGE_DETECTION,
        EMBOSS,
        IDENTITY,
        HDR_TONEMAPPING
    };

    class FilterUtils
    {
    public:
        static FilterType stringToFilterType(const std::string &filterName);
        static cv::Mat createFilterKernel(FilterType type, int kernelSize, float intensity = 1.0f);

        static void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
    };

} // namespace cuda_filter
