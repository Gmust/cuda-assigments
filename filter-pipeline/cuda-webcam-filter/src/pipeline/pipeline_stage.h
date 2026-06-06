#pragma once

#include <string>
#include <vector>

namespace cuda_filter
{
    // A single filter that can be chained in the pipeline. Convolution stages
    // carry an NxN kernel; point stages (grayscale, invert, ...) carry only a
    // scalar parameter. Stages are plain values so they are cheap to copy and
    // safe to add/remove from the pipeline at runtime.
    enum class StageKind
    {
        CONV_BLUR,
        CONV_SHARPEN,
        CONV_EDGE,
        CONV_EMBOSS,
        GRAYSCALE,
        INVERT,
        SEPIA,
        TINT
    };

    static constexpr int MAX_KSIZE = 7; // radius up to 3

    struct Stage
    {
        StageKind   kind;
        int         ksize = 3;                       // convolution stages only (odd)
        float       intensity = 1.0f;                // convolution weight scaling
        float       param = 1.2f;                    // TINT strength etc.
        float       kernel[MAX_KSIZE * MAX_KSIZE] = {0};

        bool isConv() const
        {
            return kind == StageKind::CONV_BLUR || kind == StageKind::CONV_SHARPEN ||
                   kind == StageKind::CONV_EDGE || kind == StageKind::CONV_EMBOSS;
        }
        int radius() const { return isConv() ? ksize / 2 : 0; }
        std::string name() const;
    };

    // Construct a fully-specified stage from a short name ("blur", "sharpen",
    // "edge", "emboss", "grayscale", "invert", "sepia", "tint").
    Stage makeStage(const std::string &name, int ksize = 3, float intensity = 1.0f);

    // Parse a comma-separated spec such as "blur:5,sharpen,edge" into stages.
    // Each token is "name" or "name:ksize".
    std::vector<Stage> parsePipelineSpec(const std::string &spec);

    // Human-readable summary, e.g. "blur(5) -> sharpen(3) -> edge(3)".
    std::string describePipeline(const std::vector<Stage> &stages);

} // namespace cuda_filter
