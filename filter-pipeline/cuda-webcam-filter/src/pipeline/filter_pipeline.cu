#include "filter_pipeline.h"
#include <plog/Log.h>
#include <algorithm>
#include <cstring>
#include <chrono>

namespace cuda_filter
{
    static constexpr int CHANNELS = 3;
    static constexpr int BLOCK_X  = 16;
    static constexpr int BLOCK_Y  = 16;

#define PP_CUDA(call)                                                            \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess)                                                   \
            PLOG_ERROR << "CUDA error " << #call << ": " << cudaGetErrorString(_e); \
    } while (0)

    static inline int divUp(int a, int b) { return (a + b - 1) / b; }

    // ── Band-aware kernels (buffer row 0 == global row `rowOffset`) ──────────
    __global__ void k_conv_band(const unsigned char *__restrict__ in,
                                unsigned char *__restrict__ out,
                                const float *__restrict__ ker,
                                int W, int H, int ksize,
                                int rowOffset, int y0, int y1)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = y0 + blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= W || y >= y1) return;

        int radius = ksize / 2;
        float acc[CHANNELS] = {0.f, 0.f, 0.f};
        for (int ky = -radius; ky <= radius; ++ky) {
            int gy = min(max(y + ky, 0), H - 1);
            int ly = gy - rowOffset;
            for (int kx = -radius; kx <= radius; ++kx) {
                int ix = min(max(x + kx, 0), W - 1);
                float w = ker[(ky + radius) * ksize + (kx + radius)];
                const unsigned char *p = &in[(ly * W + ix) * CHANNELS];
                acc[0] += p[0] * w; acc[1] += p[1] * w; acc[2] += p[2] * w;
            }
        }
        unsigned char *o = &out[((y - rowOffset) * W + x) * CHANNELS];
        o[0] = (unsigned char)min(max(acc[0], 0.f), 255.f);
        o[1] = (unsigned char)min(max(acc[1], 0.f), 255.f);
        o[2] = (unsigned char)min(max(acc[2], 0.f), 255.f);
    }

    __global__ void k_point_band(const unsigned char *__restrict__ in,
                                 unsigned char *__restrict__ out,
                                 int W, int H, int kind, float param,
                                 int rowOffset, int y0, int y1)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = y0 + blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= W || y >= y1) return;

        int i = ((y - rowOffset) * W + x) * CHANNELS;
        float b = in[i+0], g = in[i+1], r = in[i+2];     // OpenCV BGR
        float ob = b, og = g, orr = r;
        switch ((StageKind)kind) {
            case StageKind::GRAYSCALE: {
                float l = 0.114f*b + 0.587f*g + 0.299f*r; ob = og = orr = l;
            } break;
            case StageKind::INVERT:
                ob = 255.f-b; og = 255.f-g; orr = 255.f-r; break;
            case StageKind::SEPIA:
                orr = fminf(0.393f*r + 0.769f*g + 0.189f*b, 255.f);
                og  = fminf(0.349f*r + 0.686f*g + 0.168f*b, 255.f);
                ob  = fminf(0.272f*r + 0.534f*g + 0.131f*b, 255.f); break;
            case StageKind::TINT:
                orr = fminf(r * param, 255.f); ob = b / fmaxf(param, 0.01f); break;
            default: break;
        }
        out[i+0] = (unsigned char)ob;
        out[i+1] = (unsigned char)og;
        out[i+2] = (unsigned char)orr;
    }

    __global__ void k_wipe(const unsigned char *__restrict__ A,
                           const unsigned char *__restrict__ B,
                           unsigned char *__restrict__ out,
                           int W, int H, float line, float softness)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= W || y >= H) return;
        float wB;
        if (softness <= 0.f) wB = (x < line) ? 1.f : 0.f;
        else                 wB = fminf(fmaxf((line - x) / softness + 0.5f, 0.f), 1.f);
        int i = (y * W + x) * CHANNELS;
        out[i+0] = (unsigned char)(A[i+0]*(1.f-wB) + B[i+0]*wB);
        out[i+1] = (unsigned char)(A[i+1]*(1.f-wB) + B[i+1]*wB);
        out[i+2] = (unsigned char)(A[i+2]*(1.f-wB) + B[i+2]*wB);
    }

    static void launchStageFull(const Stage &s, float *dker,
                                const unsigned char *in, unsigned char *out,
                                int W, int H, cudaStream_t stream)
    {
        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid(divUp(W, BLOCK_X), divUp(H, BLOCK_Y));
        if (s.isConv())
            k_conv_band<<<grid, block, 0, stream>>>(in, out, dker, W, H, s.ksize, 0, 0, H);
        else
            k_point_band<<<grid, block, 0, stream>>>(in, out, W, H, (int)s.kind, s.param, 0, 0, H);
    }

    // ── lifecycle ───────────────────────────────────────────────────────────
    FilterPipeline::FilterPipeline()
    {
        PP_CUDA(cudaStreamCreate(&m_stream0));
    }

    FilterPipeline::~FilterPipeline()
    {
        freeBuffers();
        freeBands();
        freeChainDevice(m_active);
        freeChainDevice(m_incoming);
        for (auto e : m_events) cudaEventDestroy(e);
        m_events.clear();
        if (m_stream0) cudaStreamDestroy(m_stream0);
    }

    // ── chain device-kernel management ────────────────────────────────────────
    void FilterPipeline::syncChainDevice(Chain &c)
    {
        freeChainDevice(c);
        c.dKernels.resize(c.stages.size(), nullptr);
        for (size_t i = 0; i < c.stages.size(); ++i) {
            if (c.stages[i].isConv()) {
                int n = c.stages[i].ksize;
                float *dk = nullptr;
                PP_CUDA(cudaMalloc(&dk, n * n * sizeof(float)));
                PP_CUDA(cudaMemcpy(dk, c.stages[i].kernel, n*n*sizeof(float),
                                   cudaMemcpyHostToDevice));
                c.dKernels[i] = dk;
            }
        }
    }

    void FilterPipeline::freeChainDevice(Chain &c)
    {
        for (auto p : c.dKernels) if (p) cudaFree(p);
        c.dKernels.clear();
    }

    void FilterPipeline::setStages(const std::vector<Stage> &stages)
    {
        m_active.stages = stages;
        syncChainDevice(m_active);
        PLOG_INFO << "Pipeline: " << describePipeline(m_active.stages);
    }

    void FilterPipeline::addStage(const Stage &s)
    {
        m_active.stages.push_back(s);
        syncChainDevice(m_active);
        PLOG_INFO << "Pipeline: " << describePipeline(m_active.stages);
    }

    void FilterPipeline::removeLast()
    {
        if (m_active.stages.empty()) return;
        m_active.stages.pop_back();
        syncChainDevice(m_active);
        PLOG_INFO << "Pipeline: " << describePipeline(m_active.stages);
    }

    void FilterPipeline::clear()
    {
        m_active.stages.clear();
        syncChainDevice(m_active);
        PLOG_INFO << "Pipeline cleared (passthrough)";
    }

    // ── buffers ───────────────────────────────────────────────────────────────
    void FilterPipeline::ensureBuffers(int W, int H)
    {
        if (W == m_W && H == m_H && m_hIn) return;
        freeBuffers();
        freeBands();
        m_W = W; m_H = H;
        size_t bytes = (size_t)W * H * CHANNELS;
        PP_CUDA(cudaMallocHost(&m_hIn,  bytes));
        PP_CUDA(cudaMallocHost(&m_hOut, bytes));
        PP_CUDA(cudaMalloc(&m_dIn,   bytes));
        PP_CUDA(cudaMalloc(&m_dBuf[0], bytes));
        PP_CUDA(cudaMalloc(&m_dBuf[1], bytes));
        PP_CUDA(cudaMalloc(&m_dOutA, bytes));
        PP_CUDA(cudaMalloc(&m_dOutB, bytes));
        PP_CUDA(cudaMalloc(&m_dFinal, bytes));
        rebuildBands();
    }

    void FilterPipeline::freeBuffers()
    {
        if (m_hIn)  { cudaFreeHost(m_hIn);  m_hIn = nullptr; }
        if (m_hOut) { cudaFreeHost(m_hOut); m_hOut = nullptr; }
        if (m_dIn)  { cudaFree(m_dIn);  m_dIn = nullptr; }
        if (m_dBuf[0]) { cudaFree(m_dBuf[0]); m_dBuf[0] = nullptr; }
        if (m_dBuf[1]) { cudaFree(m_dBuf[1]); m_dBuf[1] = nullptr; }
        if (m_dOutA) { cudaFree(m_dOutA); m_dOutA = nullptr; }
        if (m_dOutB) { cudaFree(m_dOutB); m_dOutB = nullptr; }
        if (m_dFinal) { cudaFree(m_dFinal); m_dFinal = nullptr; }
        m_W = m_H = 0;
    }

    void FilterPipeline::rebuildBands()
    {
        freeBands();
        if (!m_dIn) return;                              // buffers not allocated yet
        // Band buffers are full-image sized so any band (+halo) fits regardless
        // of stream count or chain length — robust against runtime changes.
        size_t bytes = (size_t)m_W * m_H * CHANNELS;
        m_bands.resize(m_numStreams);
        for (auto &bb : m_bands) {
            PP_CUDA(cudaMalloc(&bb.dIn, bytes));
            PP_CUDA(cudaMalloc(&bb.dA,  bytes));
            PP_CUDA(cudaMalloc(&bb.dB,  bytes));
            PP_CUDA(cudaStreamCreate(&bb.stream));
        }
    }

    void FilterPipeline::freeBands()
    {
        for (auto &bb : m_bands) {
            if (bb.dIn) cudaFree(bb.dIn);
            if (bb.dA)  cudaFree(bb.dA);
            if (bb.dB)  cudaFree(bb.dB);
            if (bb.stream) cudaStreamDestroy(bb.stream);
        }
        m_bands.clear();
    }

    void FilterPipeline::setNumStreams(int n)
    {
        n = std::max(1, std::min(n, 16));
        if (n == m_numStreams) return;
        m_numStreams = n;
        rebuildBands();
    }

    // ── single-stream chain execution ────────────────────────────────────────
    void FilterPipeline::runChainSingle(Chain &c, unsigned char *dIn,
                                        unsigned char *dResult, int W, int H, bool timed)
    {
        size_t bytes = (size_t)W * H * CHANNELS;
        size_t n = c.stages.size();

        if (n == 0) {                                    // passthrough
            PP_CUDA(cudaMemcpyAsync(dResult, dIn, bytes, cudaMemcpyDeviceToDevice, m_stream0));
            if (timed) m_timings.clear();
            return;
        }

        if (timed) {                                     // event pool: n+1 markers
            if (m_events.size() < n + 1) {
                for (size_t i = m_events.size(); i < n + 1; ++i) {
                    cudaEvent_t e; PP_CUDA(cudaEventCreate(&e)); m_events.push_back(e);
                }
            }
            PP_CUDA(cudaEventRecord(m_events[0], m_stream0));
        }

        unsigned char *src = dIn, *dst = m_dBuf[0];
        for (size_t i = 0; i < n; ++i) {
            launchStageFull(c.stages[i], c.dKernels[i], src, dst, W, H, m_stream0);
            if (timed) PP_CUDA(cudaEventRecord(m_events[i + 1], m_stream0));
            // ping-pong: first stage reads dIn then we alternate m_dBuf[0/1]
            if (i == 0) { src = m_dBuf[0]; dst = m_dBuf[1]; }
            else        std::swap(src, dst);
        }
        // `src` now points at the final stage output.
        PP_CUDA(cudaMemcpyAsync(dResult, src, bytes, cudaMemcpyDeviceToDevice, m_stream0));

        if (timed) {
            PP_CUDA(cudaStreamSynchronize(m_stream0));
            m_timings.clear();
            for (size_t i = 0; i < n; ++i) {
                float ms = 0.f;
                cudaEventElapsedTime(&ms, m_events[i], m_events[i + 1]);
                m_timings.push_back({c.stages[i].name(), ms});
            }
        }
    }

    // ── multi-stream (row-band) execution with halo recompute ────────────────
    void FilterPipeline::runChainMulti(int W, int H)
    {
        Chain &c = m_active;
        int nBands = m_numStreams;
        int totalRadius = 0;
        for (auto &s : c.stages) totalRadius += s.radius();

        if (c.stages.empty()) {                          // passthrough
            std::memcpy(m_hOut, m_hIn, (size_t)W * H * CHANNELS);
            return;
        }

        int bandH = divUp(H, nBands);
        dim3 block(BLOCK_X, BLOCK_Y);

        for (int bi = 0; bi < nBands; ++bi) {
            int outY0 = bi * bandH;
            int outY1 = std::min(outY0 + bandH, H);
            if (outY0 >= outY1) continue;
            Band &bb = m_bands[bi];

            int upY0 = std::max(outY0 - totalRadius, 0);
            int upY1 = std::min(outY1 + totalRadius, H);
            int upRows = upY1 - upY0;

            size_t off  = (size_t)upY0 * W * CHANNELS;
            size_t span = (size_t)upRows * W * CHANNELS;
            PP_CUDA(cudaMemcpyAsync(bb.dIn, m_hIn + off, span,
                                    cudaMemcpyHostToDevice, bb.stream));

            int remaining = totalRadius;
            unsigned char *src = bb.dIn, *dst = bb.dA;
            for (size_t k = 0; k < c.stages.size(); ++k) {
                remaining -= c.stages[k].radius();
                int cY0 = std::max(outY0 - remaining, 0);
                int cY1 = std::min(outY1 + remaining, H);
                int rows = cY1 - cY0;
                dim3 grid(divUp(W, BLOCK_X), divUp(rows, BLOCK_Y));
                if (c.stages[k].isConv())
                    k_conv_band<<<grid, block, 0, bb.stream>>>(
                        src, dst, c.dKernels[k], W, H, c.stages[k].ksize, upY0, cY0, cY1);
                else
                    k_point_band<<<grid, block, 0, bb.stream>>>(
                        src, dst, W, H, (int)c.stages[k].kind, c.stages[k].param,
                        upY0, cY0, cY1);
                if (k == 0) { src = bb.dA; dst = bb.dB; }
                else        std::swap(src, dst);
            }
            unsigned char *result = src;                 // trailing swap -> latest dst
            size_t coreOff  = (size_t)(outY0 - upY0) * W * CHANNELS;
            size_t hostOff  = (size_t)outY0 * W * CHANNELS;
            size_t coreSpan = (size_t)(outY1 - outY0) * W * CHANNELS;
            PP_CUDA(cudaMemcpyAsync(m_hOut + hostOff, result + coreOff, coreSpan,
                                    cudaMemcpyDeviceToHost, bb.stream));
        }
        PP_CUDA(cudaDeviceSynchronize());
    }

    // ── transition control ────────────────────────────────────────────────────
    void FilterPipeline::beginTransition(const std::vector<Stage> &target,
                                         float durationSec, float softnessPx)
    {
        m_incoming.stages = target;
        syncChainDevice(m_incoming);
        m_trans.active   = true;
        m_trans.pos      = 0.f;
        m_trans.duration = std::max(0.05f, durationSec);
        m_trans.softness = std::max(0.f, softnessPx);
        PLOG_INFO << "Transition -> " << describePipeline(target)
                  << " over " << durationSec << "s";
    }

PP_CUDA

    // ── per-frame entry point ──────────────────────────────────────────────────
    void FilterPipeline::process(const cv::Mat &input, cv::Mat &output)
    {
        if (input.empty() || input.type() != CV_8UC3) {
            PLOG_ERROR << "Pipeline expects a non-empty CV_8UC3 frame";
            return;
        }
        int W = input.cols, H = input.rows;
        ensureBuffers(W, H);
        output.create(input.size(), input.type());
        size_t bytes = (size_t)W * H * CHANNELS;

        std::memcpy(m_hIn, input.data, bytes);

        cudaEvent_t evStart, evUp, evComp, evDown;
        cudaEventCreate(&evStart); cudaEventCreate(&evUp);
        cudaEventCreate(&evComp);  cudaEventCreate(&evDown);

        if (m_trans.active) {
            // Both chains run single-stream; wipe-composite on the GPU.
            cudaEventRecord(evStart, m_stream0);
            PP_CUDA(cudaMemcpyAsync(m_dIn, m_hIn, bytes, cudaMemcpyHostToDevice, m_stream0));
            cudaEventRecord(evUp, m_stream0);
            runChainSingle(m_active,   m_dIn, m_dOutA, W, H, /*timed*/false);
            runChainSingle(m_incoming, m_dIn, m_dOutB, W, H, /*timed*/false);
            dim3 block(BLOCK_X, BLOCK_Y), grid(divUp(W,BLOCK_X), divUp(H,BLOCK_Y));
            float line = m_trans.pos * W;
            k_wipe<<<grid, block, 0, m_stream0>>>(m_dOutA, m_dOutB, m_dFinal,
                                                  W, H, line, m_trans.softness);
            cudaEventRecord(evComp, m_stream0);
            PP_CUDA(cudaMemcpyAsync(m_hOut, m_dFinal, bytes, cudaMemcpyDeviceToHost, m_stream0));
            cudaEventRecord(evDown, m_stream0);
            PP_CUDA(cudaStreamSynchronize(m_stream0));
            m_timings.clear();
        }
        else if (m_multiStream) {
            auto t0 = std::chrono::high_resolution_clock::now();
            runChainMulti(W, H);                          // reads m_hIn, writes m_hOut
            auto t1 = std::chrono::high_resolution_clock::now();
            m_uploadMs = m_downloadMs = 0.f;
            m_computeMs = m_totalMs =
                std::chrono::duration<float, std::milli>(t1 - t0).count();
            m_timings.clear();
            m_timings.push_back({"pipeline (multi x" + std::to_string(m_numStreams) + ")",
                                 m_totalMs});
            std::memcpy(output.data, m_hOut, bytes);
            cudaEventDestroy(evStart); cudaEventDestroy(evUp);
            cudaEventDestroy(evComp);  cudaEventDestroy(evDown);
            return;
        }
        else {
            cudaEventRecord(evStart, m_stream0);
            PP_CUDA(cudaMemcpyAsync(m_dIn, m_hIn, bytes, cudaMemcpyHostToDevice, m_stream0));
            cudaEventRecord(evUp, m_stream0);
            runChainSingle(m_active, m_dIn, m_dFinal, W, H, /*timed*/true);  // syncs internally
            cudaEventRecord(evComp, m_stream0);
            PP_CUDA(cudaMemcpyAsync(m_hOut, m_dFinal, bytes, cudaMemcpyDeviceToHost, m_stream0));
            cudaEventRecord(evDown, m_stream0);
            PP_CUDA(cudaStreamSynchronize(m_stream0));
        }

        cudaEventElapsedTime(&m_uploadMs,   evStart, evUp);
        cudaEventElapsedTime(&m_computeMs,  evUp,    evComp);
        cudaEventElapsedTime(&m_downloadMs, evComp,  evDown);
        cudaEventElapsedTime(&m_totalMs,    evStart, evDown);
        cudaEventDestroy(evStart); cudaEventDestroy(evUp);
        cudaEventDestroy(evComp);  cudaEventDestroy(evDown);

        std::memcpy(output.data, m_hOut, bytes);
    }

} // namespace cuda_filter
