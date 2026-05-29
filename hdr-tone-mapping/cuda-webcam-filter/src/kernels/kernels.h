#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace cuda_filter
{

    void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);
    void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);

    struct HdrOptions
    {
        float exposure;
        float gamma;
        float saturation;
        float whitePoint;
        std::string algorithm;  // "reinhard", "aces", "local"
    };

    void applyHDRFilterGPU(const cv::Mat &input, cv::Mat &output, const HdrOptions &opts);
    void applyHDRFilterCPU(const cv::Mat &input, cv::Mat &output, const HdrOptions &opts);

    namespace cuda
    {
// CUDA-specific type declarations and helper functions
#ifdef __CUDACC__
        // These will only be visible to CUDA compiler
        __host__ __device__ inline int divUp(int a, int b)
        {
            return (a + b - 1) / b;
        }
#endif
    }

} // namespace cuda_filter