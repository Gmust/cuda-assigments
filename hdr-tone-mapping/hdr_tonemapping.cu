// Compile: nvcc -O2 -arch=native -o hdr_tonemapping.exe hdr_tonemapping.cu
// Run:     hdr_tonemapping.exe [exposure] [gamma] [saturation] [white_point]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <chrono>

static constexpr int WIDTH    = 1280;
static constexpr int HEIGHT   = 720;
static constexpr int CHANNELS = 3;

#define TILE_W  16
#define TILE_H  16
#define HALO     4
#define SH_W    (TILE_W + 2 * HALO)
#define SH_H    (TILE_H + 2 * HALO)

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

__host__ __device__ inline float rgb2lum(float r, float g, float b)
{
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

__host__ __device__ inline float clamp01(float v)
{
    return (v >= 0.0f) ? (v <= 1.0f ? v : 1.0f) : 0.0f;
}

__host__ __device__ inline unsigned char toUchar(float v)
{
    return static_cast<unsigned char>(clamp01(v) * 255.0f + 0.5f);
}

__host__ __device__ inline void satGamma(float &r, float &g, float &b,
                                          float sat, float invGamma)
{
    float lum = rgb2lum(r, g, b);
    r = lum + sat * (r - lum);
    g = lum + sat * (g - lum);
    b = lum + sat * (b - lum);
    r = powf(r < 0.0f ? 0.0f : r, invGamma);
    g = powf(g < 0.0f ? 0.0f : g, invGamma);
    b = powf(b < 0.0f ? 0.0f : b, invGamma);
}

// ACES RRT+ODT approximation (Narkowicz 2015)
__host__ __device__ inline float acesFilmic(float x)
{
    return clamp01((x * (2.51f*x + 0.03f)) / (x * (2.43f*x + 0.59f) + 0.14f));
}

// Extended Reinhard (Reinhard et al. 2002): L_out = L*(1+L/white^2)/(1+L)
__global__ void k_reinhardGlobal(const unsigned char *__restrict__ in,
                                   unsigned char       *__restrict__ out,
                                   int W, int H,
                                   float exposure, float white2,
                                   float sat,      float invGamma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i = (y * W + x) * CHANNELS;
    float r = in[i+0] / 255.0f * exposure;
    float g = in[i+1] / 255.0f * exposure;
    float b = in[i+2] / 255.0f * exposure;

    float L     = rgb2lum(r, g, b);
    float L_out = L * (1.0f + L / white2) / (1.0f + L);
    float scale = (L > 1e-6f) ? (L_out / L) : 0.0f;
    r *= scale; g *= scale; b *= scale;

    satGamma(r, g, b, sat, invGamma);
    out[i+0] = toUchar(r);
    out[i+1] = toUchar(g);
    out[i+2] = toUchar(b);
}

__global__ void k_aces(const unsigned char *__restrict__ in,
                        unsigned char       *__restrict__ out,
                        int W, int H,
                        float exposure, float sat, float invGamma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i = (y * W + x) * CHANNELS;
    float r = acesFilmic(in[i+0] / 255.0f * exposure);
    float g = acesFilmic(in[i+1] / 255.0f * exposure);
    float b = acesFilmic(in[i+2] / 255.0f * exposure);

    satGamma(r, g, b, sat, invGamma);
    out[i+0] = toUchar(r);
    out[i+1] = toUchar(g);
    out[i+2] = toUchar(b);
}

__global__ void k_toFloat(const unsigned char *__restrict__ in,
                           float               *__restrict__ out,
                           int W, int H, float exposure)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i = (y * W + x) * CHANNELS;
    out[i+0] = in[i+0] / 255.0f * exposure;
    out[i+1] = in[i+1] / 255.0f * exposure;
    out[i+2] = in[i+2] / 255.0f * exposure;
}

// 9x9 local average in shared memory (24x24 luminance tile, 2304 bytes/block)
__global__ void k_localReinhard(const float   *__restrict__ in,
                                  unsigned char *__restrict__ out,
                                  int W, int H,
                                  float sat, float invGamma)
{
    __shared__ float shLum[SH_H][SH_W];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * TILE_W;
    int by = blockIdx.y * TILE_H;

    for (int dy = ty; dy < SH_H; dy += TILE_H)
        for (int dx = tx; dx < SH_W; dx += TILE_W)
        {
            int gx = min(max(bx + dx - HALO, 0), W - 1);
            int gy = min(max(by + dy - HALO, 0), H - 1);
            int gi = (gy * W + gx) * CHANNELS;
            shLum[dy][dx] = rgb2lum(in[gi+0], in[gi+1], in[gi+2]);
        }
    __syncthreads();

    int x = bx + tx, y = by + ty;
    if (x >= W || y >= H) return;

    float sumL = 0.0f;
    #pragma unroll
    for (int dy = 0; dy < 2*HALO+1; ++dy)
        #pragma unroll
        for (int dx = 0; dx < 2*HALO+1; ++dx)
            sumL += shLum[ty + dy][tx + dx];

    float L_local = sumL / ((float)((2*HALO+1) * (2*HALO+1)));

    int gi = (y * W + x) * CHANNELS;
    float r = in[gi+0], g = in[gi+1], b = in[gi+2];
    float scale = 1.0f / (1.0f + L_local);
    r *= scale; g *= scale; b *= scale;

    satGamma(r, g, b, sat, invGamma);
    out[gi+0] = toUchar(r);
    out[gi+1] = toUchar(g);
    out[gi+2] = toUchar(b);
}

static void runGPU(const unsigned char *h_in,
                   unsigned char *h_out0, unsigned char *h_out1, unsigned char *h_out2,
                   float exposure, float white2, float sat, float invGamma,
                   float &ms0, float &ms1, float &ms2)
{
    size_t imgBytes   = (size_t)WIDTH * HEIGHT * CHANNELS;
    size_t floatBytes = imgBytes * sizeof(float);

    unsigned char *d_in, *d_out;
    float *d_float;
    CUDA_CHECK(cudaMalloc(&d_in,    imgBytes));
    CUDA_CHECK(cudaMalloc(&d_out,   imgBytes));
    CUDA_CHECK(cudaMalloc(&d_float, floatBytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, imgBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // Warmup: eliminates WDDM JIT cost from first timed run
    k_aces<<<grid, block>>>(d_in, d_out, WIDTH, HEIGHT, exposure, sat, invGamma);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(t0));
    k_reinhardGlobal<<<grid, block>>>(d_in, d_out, WIDTH, HEIGHT, exposure, white2, sat, invGamma);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms0, t0, t1));
    CUDA_CHECK(cudaMemcpy(h_out0, d_out, imgBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(t0));
    k_aces<<<grid, block>>>(d_in, d_out, WIDTH, HEIGHT, exposure, sat, invGamma);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms1, t0, t1));
    CUDA_CHECK(cudaMemcpy(h_out1, d_out, imgBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(t0));
    k_toFloat<<<grid, block>>>(d_in, d_float, WIDTH, HEIGHT, exposure);
    CUDA_CHECK(cudaGetLastError());
    k_localReinhard<<<grid, block>>>(d_float, d_out, WIDTH, HEIGHT, sat, invGamma);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms2, t0, t1));
    CUDA_CHECK(cudaMemcpy(h_out2, d_out, imgBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_float);
}

static void cpu_reinhardGlobal(const unsigned char *in, unsigned char *out,
                                float exposure, float white2, float sat, float invGamma)
{
    for (int y = 0; y < HEIGHT; ++y)
    for (int x = 0; x < WIDTH;  ++x)
    {
        int i = (y * WIDTH + x) * CHANNELS;
        float r = in[i+0] / 255.0f * exposure;
        float g = in[i+1] / 255.0f * exposure;
        float b = in[i+2] / 255.0f * exposure;
        float L = rgb2lum(r, g, b);
        float L_out = L * (1.0f + L / white2) / (1.0f + L);
        float scale = (L > 1e-6f) ? (L_out / L) : 0.0f;
        r *= scale; g *= scale; b *= scale;
        satGamma(r, g, b, sat, invGamma);
        out[i+0] = toUchar(r); out[i+1] = toUchar(g); out[i+2] = toUchar(b);
    }
}

static void cpu_aces(const unsigned char *in, unsigned char *out,
                      float exposure, float sat, float invGamma)
{
    for (int y = 0; y < HEIGHT; ++y)
    for (int x = 0; x < WIDTH;  ++x)
    {
        int i = (y * WIDTH + x) * CHANNELS;
        float r = acesFilmic(in[i+0] / 255.0f * exposure);
        float g = acesFilmic(in[i+1] / 255.0f * exposure);
        float b = acesFilmic(in[i+2] / 255.0f * exposure);
        satGamma(r, g, b, sat, invGamma);
        out[i+0] = toUchar(r); out[i+1] = toUchar(g); out[i+2] = toUchar(b);
    }
}

static void cpu_localReinhard(const unsigned char *in, unsigned char *out,
                               float exposure, float sat, float invGamma)
{
    static float fbuf[WIDTH * HEIGHT * CHANNELS];
    for (int i = 0; i < WIDTH * HEIGHT * CHANNELS; ++i)
        fbuf[i] = (in[i] / 255.0f) * exposure;

    constexpr int WINDOW = (2*HALO+1) * (2*HALO+1);

    for (int y = 0; y < HEIGHT; ++y)
    for (int x = 0; x < WIDTH;  ++x)
    {
        float sumL = 0.0f;
        for (int dy = -HALO; dy <= HALO; ++dy)
        for (int dx = -HALO; dx <= HALO; ++dx)
        {
            int ix = std::min(std::max(x+dx, 0), WIDTH-1);
            int iy = std::min(std::max(y+dy, 0), HEIGHT-1);
            int ii = (iy * WIDTH + ix) * CHANNELS;
            sumL += rgb2lum(fbuf[ii+0], fbuf[ii+1], fbuf[ii+2]);
        }
        int i = (y * WIDTH + x) * CHANNELS;
        float r = fbuf[i+0], g = fbuf[i+1], b = fbuf[i+2];
        float scale = 1.0f / (1.0f + sumL / WINDOW);
        r *= scale; g *= scale; b *= scale;
        satGamma(r, g, b, sat, invGamma);
        out[i+0] = toUchar(r); out[i+1] = toUchar(g); out[i+2] = toUchar(b);
    }
}

static void generateHDRScene(unsigned char *img)
{
    for (int y = 0; y < HEIGHT; ++y)
    for (int x = 0; x < WIDTH;  ++x)
    {
        float fx = (float)x / WIDTH;
        float fy = (float)y / HEIGHT;
        float r, g, b;

        if (fy > 0.85f)
            r = g = b = fx;
        else if (fx < 0.20f) {
            float t = 0.85f + fy * 0.25f + (float)(x % 7) * 0.01f;
            r = t * 1.10f; g = t * 0.95f; b = t * 0.60f;
        } else if (fx < 0.45f) {
            float t = 0.45f + fy * 0.35f;
            r = t * 0.55f; g = t * 0.75f; b = t * 1.00f;
        } else if (fx < 0.70f) {
            float t = 0.30f + sinf(fx * 12.0f) * 0.12f + fy * 0.20f;
            r = t * 0.60f; g = t * 0.90f; b = t * 0.45f;
        } else {
            float t = 0.03f + fy * 0.12f + (float)(y % 5) * 0.004f;
            r = t * 0.50f; g = t * 0.45f; b = t * 0.65f;
        }

        int i = (y * WIDTH + x) * CHANNELS;
        img[i+0] = toUchar(r);
        img[i+1] = toUchar(g);
        img[i+2] = toUchar(b);
    }
}

static void savePPM(const char *path, const unsigned char *img, int w, int h)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(img, 1, (size_t)w * h * CHANNELS, f);
    fclose(f);
    printf("  Saved %s\n", path);
}

int main(int argc, char **argv)
{
    float exposure   = (argc > 1) ? (float)atof(argv[1]) : 2.5f;
    float gamma      = (argc > 2) ? (float)atof(argv[2]) : 2.2f;
    float saturation = (argc > 3) ? (float)atof(argv[3]) : 1.2f;
    float whitePoint = (argc > 4) ? (float)atof(argv[4]) : 4.0f;

    float invGamma = 1.0f / gamma;
    float white2   = whitePoint * whitePoint;

    printf("=== HDR Tone Mapping Benchmark ===\n");
    printf("Image: %dx%d   exposure=%.2f  gamma=%.2f  sat=%.2f  white=%.2f\n\n",
           WIDTH, HEIGHT, exposure, gamma, saturation, whitePoint);

    size_t imgBytes = (size_t)WIDTH * HEIGHT * CHANNELS;

    unsigned char *h_in   = new unsigned char[imgBytes];
    unsigned char *h_out0 = new unsigned char[imgBytes];
    unsigned char *h_out1 = new unsigned char[imgBytes];
    unsigned char *h_out2 = new unsigned char[imgBytes];
    unsigned char *h_cpu0 = new unsigned char[imgBytes];
    unsigned char *h_cpu1 = new unsigned char[imgBytes];
    unsigned char *h_cpu2 = new unsigned char[imgBytes];

    printf("[1/4] Generating synthetic HDR scene ...\n");
    generateHDRScene(h_in);
    savePPM("hdr_input.ppm", h_in, WIDTH, HEIGHT);

    printf("[2/4] Running GPU kernels ...\n");
    float ms0 = 0, ms1 = 0, ms2 = 0;
    runGPU(h_in, h_out0, h_out1, h_out2, exposure, white2, saturation, invGamma, ms0, ms1, ms2);
    printf("  Reinhard Global : %.3f ms\n", ms0);
    printf("  ACES Filmic     : %.3f ms\n", ms1);
    printf("  Local Reinhard  : %.3f ms  (shared mem, 9x9 window)\n", ms2);
    savePPM("hdr_reinhard.ppm", h_out0, WIDTH, HEIGHT);
    savePPM("hdr_aces.ppm",     h_out1, WIDTH, HEIGHT);
    savePPM("hdr_local.ppm",    h_out2, WIDTH, HEIGHT);

    printf("[3/4] Running CPU reference ...\n");
    auto timeIt = [](auto fn) -> float {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        return std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    };
    float cpu0 = timeIt([&]{ cpu_reinhardGlobal(h_in, h_cpu0, exposure, white2, saturation, invGamma); });
    float cpu1 = timeIt([&]{ cpu_aces(h_in, h_cpu1, exposure, saturation, invGamma); });
    float cpu2 = timeIt([&]{ cpu_localReinhard(h_in, h_cpu2, exposure, saturation, invGamma); });
    printf("  Reinhard Global : %.1f ms\n", cpu0);
    printf("  ACES Filmic     : %.1f ms\n", cpu1);
    printf("  Local Reinhard  : %.1f ms  (no SIMD)\n", cpu2);

    printf("\n[4/4] Performance comparison\n");
    printf("  %-30s %8s %8s %8s\n", "Algorithm", "CPU ms", "GPU ms", "Speedup");
    printf("  %-30s %8.1f %8.3f %7.1fx\n", "Reinhard Global",         cpu0, ms0, cpu0/ms0);
    printf("  %-30s %8.1f %8.3f %7.1fx\n", "ACES Filmic",             cpu1, ms1, cpu1/ms1);
    printf("  %-30s %8.1f %8.3f %7.1fx\n", "Local Reinhard (shared)", cpu2, ms2, cpu2/ms2);

    printf("\n  Correctness (max pixel diff GPU vs CPU):\n");
    auto findMaxDiff = [&](const char *name,
                           const unsigned char *a, const unsigned char *b,
                           const unsigned char *src) {
        int m = 0, mx = 0, my = 0, mc = 0, ma = 0, mb = 0;
        for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < WIDTH;  ++x)
        for (int c = 0; c < CHANNELS; ++c) {
            int k = (y * WIDTH + x) * CHANNELS + c;
            int d = abs((int)a[k] - (int)b[k]);
            if (d > m) { m = d; mx = x; my = y; mc = c; ma = a[k]; mb = b[k]; }
        }
        constexpr const char *ch[] = {"R", "G", "B"};
        int ii = (my * WIDTH + mx) * CHANNELS;
        printf("    %-28s max diff = %3d  at (%4d,%4d) ch=%s  GPU=%3d CPU=%3d  input=(%3d,%3d,%3d)\n",
               name, m, mx, my, ch[mc], ma, mb,
               (int)src[ii], (int)src[ii+1], (int)src[ii+2]);
    };
    findMaxDiff("Reinhard Global", h_out0, h_cpu0, h_in);
    findMaxDiff("ACES Filmic",     h_out1, h_cpu1, h_in);
    findMaxDiff("Local Reinhard",  h_out2, h_cpu2, h_in);
    printf("  (Diff<=1 is FP rounding; larger values indicate a bug)\n");

    delete[] h_in;
    delete[] h_out0; delete[] h_out1; delete[] h_out2;
    delete[] h_cpu0; delete[] h_cpu1; delete[] h_cpu2;
    return 0;
}
