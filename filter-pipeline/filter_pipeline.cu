// Compile: nvcc -O2 -arch=native -o filter_pipeline.exe filter_pipeline.cu
// Run:     filter_pipeline.exe            (full sweep -> pipeline_benchmark.csv)
//          filter_pipeline.exe --quick    (single config, fast smoke test)
//
// Standalone, OpenCV-free reference for the CUDA Filter Pipeline assignment.
// It mirrors the kernels and execution strategies used by the integrated
// cuda-webcam-filter app, so the benchmark numbers in README.md can be
// reproduced with nothing but nvcc.
//
// What it demonstrates:
//   * A device-resident, ping-pong filter pipeline (no host round-trip between
//     stages -> intermediate results never leave the GPU).
//   * Single-stream execution (the obvious implementation).
//   * Multi-stream execution: the frame is split into horizontal bands, one
//     CUDA stream per band, so the H2D copy / compute / D2H copy of different
//     bands overlap. Chained-filter correctness across band seams is handled
//     with the overlapping-halo recompute technique.
//   * A left-to-right wipe transition between two pipelines.
//   * Per-configuration timing + a single-vs-multi-stream correctness check.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

static constexpr int CHANNELS  = 3;
static constexpr int MAX_KSIZE = 7;     // radius up to 3
static constexpr int BLOCK_X   = 16;
static constexpr int BLOCK_Y   = 16;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d - %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────
//  Stage description
// ─────────────────────────────────────────────────────────────────────────
enum class StageKind { CONV, GRAYSCALE, INVERT, SEPIA, TINT };

struct Stage {
    StageKind kind;
    int       ksize  = 1;          // CONV only (odd, <= MAX_KSIZE)
    float     param  = 1.0f;       // TINT strength, etc.
    float     kernel[MAX_KSIZE * MAX_KSIZE] = {0};

    eKind::CONV) ? ksize / 2 : 0; }
};

// Build the standard convolution kernels used by the app.
static Stage makeConv(const char *name, int ksize, float intensity)
{
    Stage s; s.kind = StageKind::CONV; s.ksize = ksize;
    float *k = s.kernel;
    int n = ksize, c = ksize / 2;
    auto at = [&](int r, int col) -> float& { return k[r * n + col]; };

    if (!strcmp(name, "blur")) {
        for (int i = 0; i < n * n; ++i) k[i] = 1.0f / (n * n);
    } else if (!strcmp(name, "sharpen")) {
        at(c, c) = 1.0f + 4.0f * intensity;
        if (n >= 3) { at(c-1,c) = at(c+1,c) = at(c,c-1) = at(c,c+1) = -intensity; }
    } else if (!strcmp(name, "edge")) {
        if (n >= 3) {
            at(0,0)=at(0,1)=at(0,2)=-intensity;
            at(1,0)=-intensity; at(1,1)=8.0f*intensity; at(1,2)=-intensity;
            at(2,0)=at(2,1)=at(2,2)=-intensity;
        } else { at(c,c) = 1.0f; }
    } else if (!strcmp(name, "emboss")) {
        if (n >= 3) {
            at(0,0)=-2.0f*intensity; at(0,1)=-intensity; at(0,2)=0;
            at(1,0)=-intensity;      at(1,1)=1.0f;       at(1,2)=intensity;
            at(2,0)=0;               at(2,1)=intensity;  at(2,2)=2.0f*intensity;
        } else { at(c,c) = 1.0f; }
    } else { // identity
        at(c, c) = 1.0f;
    }
    return s;
}

// ─────────────────────────────────────────────────────────────────────────
//  Band-aware kernels
//
//  Buffers are "band-local": row 0 of the buffer corresponds to global image
//  row `rowOffset`. Threads compute global coordinates and the kernel maps back
//  into the band buffer. For the single-stream full-image path, rowOffset = 0
//  and the band buffer is the whole image, so the exact same kernels are reused.
// ─────────────────────────────────────────────────────────────────────────
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
        int ly = gy - rowOffset;                       // band-local row
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
    float b = in[i+0], g = in[i+1], r = in[i+2];        // OpenCV BGR order
    float ob = b, og = g, orr = r;

    switch (kind) {
        case (int)StageKind::GRAYSCALE: {
            float l = 0.114f*b + 0.587f*g + 0.299f*r;
            ob = og = orr = l;
        } break;
        case (int)StageKind::INVERT:
            ob = 255.f - b; og = 255.f - g; orr = 255.f - r; break;
        case (int)StageKind::SEPIA: {
            orr = fminf(0.393f*r + 0.769f*g + 0.189f*b, 255.f);
            og  = fminf(0.349f*r + 0.686f*g + 0.168f*b, 255.f);
            ob  = fminf(0.272f*r + 0.534f*g + 0.131f*b, 255.f);
        } break;
        case (int)StageKind::TINT:
            orr = fminf(r * param, 255.f); ob = b / fmaxf(param, 0.01f); break;
    }
    out[i+0] = (unsigned char)ob;
    out[i+1] = (unsigned char)og;
    out[i+2] = (unsigned char)orr;
}

// Left-to-right wipe: pixels left of the wipe line show B (the incoming
// pipeline), pixels to the right show A (the outgoing pipeline). `softness`
// (in pixels) feathers the seam.
__global__ void k_wipe(const unsigned char *__restrict__ A,
                       const unsigned char *__restrict__ B,
                       unsigned char *__restrict__ out,
                       int W, int H, float line, float softness)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float wB;                                            // weight of incoming B
    if (softness <= 0.f) wB = (x < line) ? 1.f : 0.f;
    else                 wB = fminf(fmaxf((line - x) / softness + 0.5f, 0.f), 1.f);

    int i = (y * W + x) * CHANNELS;
    out[i+0] = (unsigned char)(A[i+0] * (1.f - wB) + B[i+0] * wB);
    out[i+1] = (unsigned char)(A[i+1] * (1.f - wB) + B[i+1] * wB);
    out[i+2] = (unsigned char)(A[i+2] * (1.f - wB) + B[i+2] * wB);
}

static inline int divUp(int a, int b) { return (a + b - 1) / b; }

// ─────────────────────────────────────────────────────────────────────────
//  Single-stream pipeline: one H2D, stages run full-image on stream 0, one D2H.
// ─────────────────────────────────────────────────────────────────────────
struct DeviceConst {                                     // stage kernels on device
    std::vector<float*> d_kernels;
    void upload(const std::vector<Stage> &p) {
        for (auto &s : p) {
            float *dk = nullptr;
            if (s.kind == StageKind::CONV) {
                CUDA_CHECK(cudaMalloc(&dk, s.ksize * s.ksize * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(dk, s.kernel, s.ksize*s.ksize*sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            d_kernels.push_back(dk);
        }
    }
    void free() { for (auto p : d_kernels) if (p) cudaFree(p); d_kernels.clear(); }
};

static void launchStageFull(const Stage &s, float *dker,
                            const unsigned char *in, unsigned char *out,
                            int W, int H, cudaStream_t stream)
{
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(divUp(W, BLOCK_X), divUp(H, BLOCK_Y));
    if (s.kind == StageKind::CONV)
        k_conv_band<<<grid, block, 0, stream>>>(in, out, dker, W, H, s.ksize, 0, 0, H);
    else
        k_point_band<<<grid, block, 0, stream>>>(in, out, W, H, (int)s.kind, s.param, 0, 0, H);
}

static void runSingleStream(const std::vector<Stage> &p, const DeviceConst &dc,
                            const unsigned char *h_in, unsigned char *h_out,
                            unsigned char *d_a, unsigned char *d_b,
                            int W, int H, cudaStream_t stream)
{
    size_t bytes = (size_t)W * H * CHANNELS;
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_in, bytes, cudaMemcpyHostToDevice, stream));

    unsigned char *src = d_a, *dst = d_b;
    for (size_t i = 0; i < p.size(); ++i) {
        launchStageFull(p[i], dc.d_kernels[i], src, dst, W, H, stream);
        std::swap(src, dst);
    }
    CUDA_CHECK(cudaMemcpyAsync(h_out, src, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ─────────────────────────────────────────────────────────────────────────
//  Multi-stream pipeline: N horizontal bands, one stream each. Overlapping-halo
//  recompute keeps chained filters seam-free; per-band async copies overlap.
// ─────────────────────────────────────────────────────────────────────────
struct BandBuffers {
    unsigned char *d_in = nullptr;        // band input (with halo), row0 == upY0
    unsigned char *d_a  = nullptr;        // ping
    unsigned char *d_b  = nullptr;        // pong
    cudaStream_t   stream{};
};

static void runMultiStream(const std::vector<Stage> &p, const DeviceConst &dc,
                           const unsigned char *h_in, unsigned char *h_out,
                           std::vector<BandBuffers> &bands, int nBands,
                           int W, int H)
{
    int totalRadius = 0;
    for (auto &s : p) totalRadius += s.radius();

    int bandH = divUp(H, nBands);
    dim3 block(BLOCK_X, BLOCK_Y);

    for (int bi = 0; bi < nBands; ++bi) {
        int outY0 = bi * bandH;
        int outY1 = std::min(outY0 + bandH, H);
        if (outY0 >= outY1) continue;
        BandBuffers &bb = bands[bi];

        int upY0 = std::max(outY0 - totalRadius, 0);
        int upY1 = std::min(outY1 + totalRadius, H);
        int upRows = upY1 - upY0;

        // H2D: just this band's input slice (+halo), async on the band's stream.
        size_t off  = (size_t)upY0 * W * CHANNELS;
        size_t span = (size_t)upRows * W * CHANNELS;
        CUDA_CHECK(cudaMemcpyAsync(bb.d_in, h_in + off, span,
                                   cudaMemcpyHostToDevice, bb.stream));

        // Stage k must produce rows [outY0 - H_{k+1}, outY1 + H_{k+1}) where
        // H_{k+1} is the summed radius of the *remaining* stages. The halo
        // shrinks toward zero so the last stage produces exactly the core band.
        int remaining = totalRadius;
        unsigned char *src = bb.d_in, *dst = bb.d_a;

        for (size_t k = 0; k < p.size(); ++k) {
            remaining -= p[k].radius();
            int cY0 = std::max(outY0 - remaining, 0);
            int cY1 = std::min(outY1 + remaining, H);
            int rows = cY1 - cY0;
            dim3 grid(divUp(W, BLOCK_X), divUp(rows, BLOCK_Y));

            if (p[k].kind == StageKind::CONV)
                k_conv_band<<<grid, block, 0, bb.stream>>>(
                    src, dst, dc.d_kernels[k], W, H, p[k].ksize, upY0, cY0, cY1);
            else
                k_point_band<<<grid, block, 0, bb.stream>>>(
                    src, dst, W, H, (int)p[k].kind, p[k].param, upY0, cY0, cY1);

            // First stage reads d_in; afterwards ping-pong between d_a / d_b.
            if (k == 0) { src = bb.d_a; dst = bb.d_b; }
            else        { std::swap(src, dst); }
        }
        // The loop's trailing swap leaves `src` pointing at the most recent
        // destination, i.e. the final stage output for this band.
        unsigned char *result = src;

        // D2H: only the core rows of this band.
        size_t coreOff  = (size_t)(outY0 - upY0) * W * CHANNELS;
        size_t hostOff  = (size_t)outY0 * W * CHANNELS;
        size_t coreSpan = (size_t)(outY1 - outY0) * W * CHANNELS;
        CUDA_CHECK(cudaMemcpyAsync(h_out + hostOff, result + coreOff, coreSpan,
                                   cudaMemcpyDeviceToHost, bb.stream));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ─────────────────────────────────────────────────────────────────────────
//  Host helpers
// ─────────────────────────────────────────────────────────────────────────
static void generateScene(unsigned char *img, int W, int H)
{
    for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
        float fx = (float)x / W, fy = (float)y / H;
        int i = (y * W + x) * CHANNELS;
        img[i+0] = (unsigned char)(127.5f * (1.f + sinf(fx * 18.f)) * (0.4f + 0.6f*fy));
        img[i+1] = (unsigned char)(127.5f * (1.f + sinf((fx+fy) * 12.f)));
        img[i+2] = (unsigned char)(255.f * fx * (1.f - 0.5f*fy) + ((x^y) & 31));
    }
}

static void savePPM(const char *path, const unsigned char *bgr, int W, int H)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    std::vector<unsigned char> rgb((size_t)W * H * 3);
    for (int i = 0; i < W * H; ++i) {            // BGR -> RGB for the PPM viewer
        rgb[i*3+0] = bgr[i*3+2];
        rgb[i*3+1] = bgr[i*3+1];
        rgb[i*3+2] = bgr[i*3+0];
    }
    fwrite(rgb.data(), 1, rgb.size(), f);
    fclose(f);
}

static std::vector<Stage> makePipeline(int len)
{
    std::vector<Stage> p;
    const char *chain[] = {"blur", "sharpen", "edge", "emboss", "blur"};
    for (int i = 0; i < len; ++i) p.push_back(makeConv(chain[i % 5], 3, 1.0f));
    return p;
}

// Time `iters` runs of `fn` (ms), discarding `warmup` initial runs.
template <class F>
static double timeMs(F fn, int iters, int warmup)
{
    for (int i = 0; i < warmup; ++i) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

struct Buffers {
    unsigned char *h_in = nullptr, *h_out = nullptr, *h_ref = nullptr;
    unsigned char *d_a = nullptr, *d_b = nullptr;        // single-stream full buffers
    std::vector<BandBuffers> bands;
    int W = 0, H = 0, allocStreams = 0;

    void alloc(int w, int h, int maxStreams, int minStreams, int maxRadius) {
        W = w; H = h; allocStreams = maxStreams;
        size_t bytes = (size_t)W * H * CHANNELS;
        CUDA_CHECK(cudaMallocHost(&h_in,  bytes));
        CUDA_CHECK(cudaMallocHost(&h_out, bytes));
        h_ref = new unsigned char[bytes];
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));

        // The largest band occurs with the fewest streams; size buffers for it.
        int bandH = divUp(H, minStreams);
        int maxBandRows = bandH + 2 * maxRadius;
        size_t bandBytes = (size_t)maxBandRows * W * CHANNELS;
        bands.resize(maxStreams);
        for (auto &bb : bands) {
            CUDA_CHECK(cudaMalloc(&bb.d_in, bandBytes));
            CUDA_CHECK(cudaMalloc(&bb.d_a,  bandBytes));
            CUDA_CHECK(cudaMalloc(&bb.d_b,  bandBytes));
            CUDA_CHECK(cudaStreamCreate(&bb.stream));
        }
    }
    void free() {
        cudaFreeHost(h_in); cudaFreeHost(h_out); delete[] h_ref;
        cudaFree(d_a); cudaFree(d_b);
        for (auto &bb : bands) {
            cudaFree(bb.d_in); cudaFree(bb.d_a); cudaFree(bb.d_b);
            cudaStreamDestroy(bb.stream);
        }
        bands.clear();
    }
};

static int maxPixelDiff(const unsigned char *a, const unsigned char *b, size_t n)
{
    int m = 0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, abs((int)a[i] - (int)b[i]));
    return m;
}

int main(int argc, char **argv)
{
    bool quick = (argc > 1 && !strcmp(argv[1], "--quick"));

    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== CUDA Filter Pipeline Benchmark ===\n");
    printf("GPU: %s  (%d SMs, CC %d.%d)\n\n", prop.name, prop.multiProcessorCount,
           prop.major, prop.minor);

    struct Res { int w, h; const char *name; };
    std::vector<Res> resolutions = quick
        ? std::vector<Res>{ {1280, 720, "720p"} }
        : std::vector<Res>{ {640,480,"480p"}, {1280,720,"720p"}, {1920,1080,"1080p"} };
    std::vector<int> pipeLens   = quick ? std::vector<int>{3} : std::vector<int>{1, 3, 5};
    std::vector<int> streamSet  = quick ? std::vector<int>{4} : std::vector<int>{2, 4, 8};
    const int iters = quick ? 50 : 200;
    const int warm  = 20;

    FILE *csv = fopen("pipeline_benchmark.csv", "w");
    fprintf(csv, "resolution,width,height,pipeline_len,mode,streams,ms,fps\n");

    printf("%-7s %5s %-6s %8s %9s %8s\n",
           "Res", "Len", "Mode", "Streams", "ms/frame", "FPS");
    printf("------------------------------------------------------------\n");

    for (auto &r : resolutions) {
        for (int len : pipeLens) {
            std::vector<Stage> p = makePipeline(len);
            int totalRadius = 0; for (auto &s : p) totalRadius += s.radius();

            int maxStreams = *std::max_element(streamSet.begin(), streamSet.end());
            int minStreams = *std::min_element(streamSet.begin(), streamSet.end());
            Buffers buf; buf.alloc(r.w, r.h, maxStreams, minStreams, totalRadius);
            generateScene(buf.h_in, r.w, r.h);
            DeviceConst dc; dc.upload(p);

            // --- single stream baseline ---
            cudaStream_t s0; CUDA_CHECK(cudaStreamCreate(&s0));
            double ms1 = timeMs([&]{
                runSingleStream(p, dc, buf.h_in, buf.h_out, buf.d_a, buf.d_b,
                                r.w, r.h, s0);
            }, iters, warm);
            memcpy(buf.h_ref, buf.h_out, (size_t)r.w*r.h*CHANNELS);   // reference
            fprintf(csv, "%s,%d,%d,%d,single,1,%.4f,%.1f\n",
                    r.name, r.w, r.h, len, ms1, 1000.0/ms1);
            printf("%-7s %5d %-6s %8d %9.4f %8.1f\n",
                   r.name, len, "single", 1, ms1, 1000.0/ms1);

            // --- multi stream sweep ---
            for (int ns : streamSet) {
                double msN = timeMs([&]{
                    runMultiStream(p, dc, buf.h_in, buf.h_out, buf.bands, ns,
                                   r.w, r.h);
                }, iters, warm);
                int diff = maxPixelDiff(buf.h_ref, buf.h_out,
                                        (size_t)r.w*r.h*CHANNELS);
                fprintf(csv, "%s,%d,%d,%d,multi,%d,%.4f,%.1f\n",
                        r.name, r.w, r.h, len, ns, msN, 1000.0/msN);
                printf("%-7s %5d %-6s %8d %9.4f %8.1f   (vs single: %.2fx, maxdiff=%d)\n",
                       r.name, len, "multi", ns, msN, 1000.0/msN, ms1/msN, diff);
            }

            // Visual outputs + wipe transition demo for the headline config.
            if (len == 3 && r.w == 1280) {
                savePPM("pipeline_input.ppm", buf.h_in, r.w, r.h);
                savePPM("pipeline_output.ppm", buf.h_ref, r.w, r.h);

                std::vector<Stage> pB = { makeConv("identity",3,1),  };
                pB[0].kind = StageKind::SEPIA;                 // incoming pipeline
                DeviceConst dcB; dcB.upload(pB);
                runSingleStream(pB, dcB, buf.h_in, buf.h_out, buf.d_a, buf.d_b,
                                r.w, r.h, s0);
                // composite a 60%% wipe of B (sepia) over A (blur->sharpen->edge)
                size_t bytes = (size_t)r.w*r.h*CHANNELS;
                unsigned char *dA, *dB, *dO;
                CUDA_CHECK(cudaMalloc(&dA, bytes)); CUDA_CHECK(cudaMalloc(&dB, bytes));
                CUDA_CHECK(cudaMalloc(&dO, bytes));
                CUDA_CHECK(cudaMemcpy(dA, buf.h_ref, bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(dB, buf.h_out, bytes, cudaMemcpyHostToDevice));
                dim3 block(BLOCK_X, BLOCK_Y), grid(divUp(r.w,BLOCK_X), divUp(r.h,BLOCK_Y));
                k_wipe<<<grid, block>>>(dA, dB, dO, r.w, r.h, 0.6f*r.w, 40.f);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(buf.h_out, dO, bytes, cudaMemcpyDeviceToHost));
                savePPM("pipeline_wipe.ppm", buf.h_out, r.w, r.h);
                cudaFree(dA); cudaFree(dB); cudaFree(dO); dcB.free();
                printf("   wrote pipeline_input.ppm / _output.ppm / _wipe.ppm\n");
            }

            cudaStreamDestroy(s0);
            dc.free();
            buf.free();
        }
    }

    fclose(csv);
    printf("\nWrote pipeline_benchmark.csv\n");
    return 0;
}
