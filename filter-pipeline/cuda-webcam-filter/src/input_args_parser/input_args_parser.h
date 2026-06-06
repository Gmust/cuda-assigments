#pragma once

#include <string>

// Forward-declare instead of including <cxxopts.hpp> here: this header is pulled
// into main.cpp, which is compiled as CUDA (-x cu), and nvcc's device front-end
// rejects cxxopts' __declspec(selectany) globals. The real include lives in the
// .cpp, the only translation unit that actually uses cxxopts.
namespace cxxopts { class Options; }

namespace cuda_filter
{

    enum class InputSource
    {
        WEBCAM,
        IMAGE,
        VIDEO,
        SYNTHETIC
    };

    enum class SyntheticPattern
    {
        CHECKERBOARD,
        GRADIENT,
        NOISE
    };

    struct FilterOptions
    {
        InputSource inputSource;
        std::string inputPath;
        SyntheticPattern syntheticPattern;
        int deviceId;
        std::string filterType;
        int kernelSize;
        float sigma;
        float intensity;
        bool preview;
        // HDR tone mapping parameters
        float exposure;
        float gamma;
        float saturation;
        float whitePoint;
        std::string hdrAlgorithm;
        // Filter pipeline parameters
        std::string pipelineSpec;       // e.g. "blur:5,sharpen,edge"; empty = single-filter mode
        bool        multiStream;        // run the pipeline across N CUDA streams
        int         numStreams;         // number of band streams for multi-stream mode
        std::string transitionSpec;     // target chain for the wipe transition
        float       transitionDuration; // seconds for a full left-to-right wipe
        float       wipeSoftness;       // feather width of the wipe seam, in pixels
    };

    class InputArgsParser
    {
    public:
        InputArgsParser(int argc, char **argv);

        FilterOptions parseArgs();

    private:
        int m_argc;
        char **m_argv;

        void setupOptions(cxxopts::Options &options);
        InputSource stringToInputSource(const std::string &str);
        SyntheticPattern stringToSyntheticPattern(const std::string &str);
    };

} // namespace cuda_filter
