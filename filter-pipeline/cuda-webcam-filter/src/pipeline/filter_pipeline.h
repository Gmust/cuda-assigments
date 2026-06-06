#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include "pipeline_stage.h"

namespace cuda_filter
{
    struct StageTiming
    {
        std::string name;
        float       ms;
    };

    // GPU filter pipeline with a device-resident ping-pong buffer (intermediate
    // results never return to the host between stages), runtime add/remove of
    // stages, optional multi-stream (row-band) execution, a left-to-right wipe
    // transition between two chains, and per-stage timing instrumentation.
    class FilterPipeline
    {
    public:
        FilterPipeline();
        ~FilterPipeline();

        // ── chain management (safe to call every frame / from key handlers) ──
        void setStages(const std::vector<Stage> &stages);
        void addStage(const Stage &s);
        void removeLast();
        void clear();
        const std::vector<Stage> &stages() const { return m_active.stages; }

        // ── CUDA stream configuration ──
        void setMultiStream(bool on) { m_multiStream = on; }
        bool multiStream() const { return m_multiStream; }
        void setNumStreams(int n);
        int  numStreams() const { return m_numStreams; }

        // ── wipe transition ──
        // Wipes the current chain out (to the right) while `target` wipes in
        // (from the left) over `durationSec`. `softnessPx` feathers the seam.
        void beginTransition(const std::vector<Stage> &target,
                             float durationSec, float softnessPx);
        void update(float dtSec);                 // advance the transition clock
        bool  transitioning() const { return m_trans.active; }
        float transitionProgress() const { return m_trans.pos; }

        // ── per-frame processing (input/output are BGR CV_8UC3) ──
        void process(const cv::Mat &input, cv::Mat &output);

        // ── instrumentation (valid after process()) ──
        const std::vector<StageTiming> &lastStageTimings() const { return m_timings; }
        float lastTotalMs() const { return m_totalMs; }
        float lastUploadMs() const { return m_uploadMs; }
        float lastComputeMs() const { return m_computeMs; }
        float lastDownloadMs() const { return m_downloadMs; }

    private:
        struct Chain
        {
            std::vector<Stage>  stages;
            std::vector<float*> dKernels;          // device kernel weights (null = point filter)
        };
        struct Band
        {
            unsigned char *dIn = nullptr, *dA = nullptr, *dB = nullptr;
            cudaStream_t   stream{};
        };

        void syncChainDevice(Chain &c);
        void freeChainDevice(Chain &c);
        void ensureBuffers(int W, int H);
        void freeBuffers();
        void rebuildBands();
        void freeBands();

        // Run a chain single-stream from dIn into dResult (device->device).
        // When `timed`, fills m_timings with per-stage GPU times.
        void runChainSingle(Chain &c, unsigned char *dIn, unsigned char *dResult,
                            int W, int H, bool timed);
        // Run the active chain across N band streams; reads m_hIn, writes m_hOut.
        void runChainMulti(int W, int H);

        Chain m_active;
        Chain m_incoming;

        struct Trans { bool active=false; float pos=0.f; float duration=1.f; float softness=40.f; } m_trans;

        bool m_multiStream = false;
        int  m_numStreams  = 4;

        int m_W = 0, m_H = 0;
        unsigned char *m_hIn = nullptr, *m_hOut = nullptr;            // pinned host
        unsigned char *m_dIn = nullptr;
        unsigned char *m_dBuf[2] = {nullptr, nullptr};               // ping-pong
        unsigned char *m_dOutA = nullptr, *m_dOutB = nullptr, *m_dFinal = nullptr;
        cudaStream_t   m_stream0{};
        std::vector<Band>   m_bands;
        std::vector<cudaEvent_t> m_events;                           // timing pool

        std::vector<StageTiming> m_timings;
        float m_totalMs = 0, m_uploadMs = 0, m_computeMs = 0, m_downloadMs = 0;
    };

} // namespace cuda_filter
