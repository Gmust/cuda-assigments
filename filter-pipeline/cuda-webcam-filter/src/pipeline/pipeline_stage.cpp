#include "pipeline_stage.h"
#include <plog/Log.h>
#include <cstring>
#include <sstream>

namespace cuda_filter
{
    std::string Stage::name() const
    {
        switch (kind)
        {
        case StageKind::CONV_BLUR:    return "blur";
        case StageKind::CONV_SHARPEN: return "sharpen";
        case StageKind::CONV_EDGE:    return "edge";
        case StageKind::CONV_EMBOSS:  return "emboss";
        case StageKind::GRAYSCALE:    return "grayscale";
        case StageKind::INVERT:       return "invert";
        case StageKind::SEPIA:        return "sepia";
        case StageKind::TINT:         return "tint";
        }
        return "?";
    }

    static void buildConvKernel(Stage &s)
    {
        int n = s.ksize, c = n / 2;
        float intensity = s.intensity;
        float *k = s.kernel;
        std::memset(k, 0, sizeof(s.kernel));
        auto at = [&](int r, int col) -> float & { return k[r * n + col]; };

        switch (s.kind)
        {
        case StageKind::CONV_BLUR:
            for (int i = 0; i < n * n; ++i) k[i] = 1.0f / (n * n);
            break;
        case StageKind::CONV_SHARPEN:
            at(c, c) = 1.0f + 4.0f * intensity;
            if (n >= 3) { at(c-1,c) = at(c+1,c) = at(c,c-1) = at(c,c+1) = -intensity; }
            break;
        case StageKind::CONV_EDGE:
            if (n >= 3) {
                at(0,0)=at(0,1)=at(0,2)=-intensity;
                at(1,0)=-intensity; at(1,1)=8.0f*intensity; at(1,2)=-intensity;
                at(2,0)=at(2,1)=at(2,2)=-intensity;
            } else { at(c,c) = 1.0f; }
            break;
        case StageKind::CONV_EMBOSS:
            if (n >= 3) {
                at(0,0)=-2.0f*intensity; at(0,1)=-intensity; at(0,2)=0;
                at(1,0)=-intensity;      at(1,1)=1.0f;       at(1,2)=intensity;
                at(2,0)=0;               at(2,1)=intensity;  at(2,2)=2.0f*intensity;
            } else { at(c,c) = 1.0f; }
            break;
        default:
            break;
        }
    }

    Stage makeStage(const std::string &name, int ksize, float intensity)
    {
        Stage s;
        s.ksize = (ksize % 2 == 0) ? ksize + 1 : ksize;
        if (s.ksize > MAX_KSIZE) s.ksize = MAX_KSIZE;
        if (s.ksize < 1) s.ksize = 1;
        s.intensity = intensity;

        if      (name == "blur")      s.kind = StageKind::CONV_BLUR;
        else if (name == "sharpen")   s.kind = StageKind::CONV_SHARPEN;
        else if (name == "edge")      s.kind = StageKind::CONV_EDGE;
        else if (name == "emboss")    s.kind = StageKind::CONV_EMBOSS;
        else if (name == "grayscale" || name == "gray") s.kind = StageKind::GRAYSCALE;
        else if (name == "invert")    s.kind = StageKind::INVERT;
        else if (name == "sepia")     s.kind = StageKind::SEPIA;
        else if (name == "tint")      s.kind = StageKind::TINT;
        else {
            PLOG_WARNING << "Unknown filter '" << name << "', using blur";
            s.kind = StageKind::CONV_BLUR;
        }

        if (s.isConv()) buildConvKernel(s);
        return s;
    }

    std::vector<Stage> parsePipelineSpec(const std::string &spec)
    {
        std::vector<Stage> stages;
        std::stringstream ss(spec);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            if (token.empty()) continue;
            std::string name = token;
            int ksize = 3;
            auto colon = token.find(':');
            if (colon != std::string::npos)
            {
                name = token.substr(0, colon);
                try { ksize = std::stoi(token.substr(colon + 1)); }
                catch (...) { ksize = 3; }
            }
            stages.push_back(makeStage(name, ksize));
        }
        return stages;
    }

    std::string describePipeline(const std::vector<Stage> &stages)
    {
        if (stages.empty()) return "(passthrough)";
        std::string out;
        for (size_t i = 0; i < stages.size(); ++i)
        {
            if (i) out += " -> ";
            out += stages[i].name();
            if (stages[i].isConv()) out += "(" + std::to_string(stages[i].ksize) + ")";
        }
        return out;
    }

} // namespace cuda_filter
