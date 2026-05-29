#include "kernels.h"
#include <cuda_runtime.h>
#include <plog/Log.h>
#include <vector>
#include <algorithm>

namespace cuda_filter
{

#define CHECK_CUDA_ERROR(call)                                                          \
    {                                                                                   \
        cudaError_t err = call;                                                         \
        if (err != cudaSuccess)                                                         \
        {                                                                               \
            PLOG_ERROR << "CUDA error in " << #call << ": " << cudaGetErrorString(err); \
            return;                                                                     \
        }                                                                               \
    }

    __global__ void convolutionKernel(const unsigned char *input, unsigned char *output,
                                      const float *kernel, int width, int height,
                                      int channels, int kernelSize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int radius = kernelSize / 2;

        for (int c = 0; c < channels; c++)
        {
            float sum = 0.0f;

            for (int ky = -radius; ky <= radius; ky++)
            {
                for (int kx = -radius; kx <= radius; kx++)
                {
                    int ix = min(max(x + kx, 0), width - 1);
                    int iy = min(max(y + ky, 0), height - 1);

                    float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];
                    float pixelValue = input[(iy * width + ix) * channels + c];

                    sum += pixelValue * kernelValue;
                }
            }

            output[(y * width + x) * channels + c] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
        }
    }

    void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        output.create(input.size(), input.type());

        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;

        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;
        float *d_kernel = nullptr;

        size_t imageSize = width * height * channels * sizeof(unsigned char);
        size_t kernelSize_bytes = kernelSize * kernelSize * sizeof(float);

        float *h_kernel = new float[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize; i++)
            for (int j = 0; j < kernelSize; j++)
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);

        CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernelSize_bytes));

        CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernelSize_bytes, cudaMemcpyHostToDevice));

        dim3 blockDim(16, 16);
        dim3 gridDim(cuda::divUp(width, blockDim.x), cuda::divUp(height, blockDim.y));

        convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, width, height, channels, kernelSize);

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);
        delete[] h_kernel;
    }

    void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        output.create(input.size(), input.type());

        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;
        int radius = kernelSize / 2;

        float *h_kernel = new float[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize; i++)
            for (int j = 0; j < kernelSize; j++)
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    float sum = 0.0f;

                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            int ix = std::min(std::max(x + kx, 0), width - 1);
                            int iy = std::min(std::max(y + ky, 0), height - 1);

                            float kernelValue = h_kernel[(ky + radius) * kernelSize + (kx + radius)];
                            float pixelValue = input.at<cv::Vec3b>(iy, ix)[c];

                            sum += pixelValue * kernelValue;
                        }
                    }

                    output.at<cv::Vec3b>(y, x)[c] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
                }
            }
        }

        delete[] h_kernel;
    }


#define HDR_TILE_W 16
#define HDR_TILE_H 16
#define HDR_HALO   4
#define HDR_SH_W   (HDR_TILE_W + 2 * HDR_HALO)
#define HDR_SH_H   (HDR_TILE_H + 2 * HDR_HALO)

// OpenCV is BGR: channel 0=B, 1=G, 2=R — Rec. 709 coefficients applied accordingly
__host__ __device__ static inline float hdr_bgr2lum(float b, float g, float r)
{
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

__host__ __device__ static inline float hdr_clamp01(float v)
{
    return (v >= 0.0f) ? (v <= 1.0f ? v : 1.0f) : 0.0f;
}

__host__ __device__ static inline unsigned char hdr_toUchar(float v)
{
    return static_cast<unsigned char>(hdr_clamp01(v) * 255.0f + 0.5f);
}

__host__ __device__ static inline void hdr_satGamma(float &c0, float &c1, float &c2,
                                                     float sat, float invGamma)
{
    float lum = hdr_bgr2lum(c0, c1, c2);
    c0 = lum + sat * (c0 - lum);
    c1 = lum + sat * (c1 - lum);
    c2 = lum + sat * (c2 - lum);
    c0 = powf(c0 < 0.0f ? 0.0f : c0, invGamma);
    c1 = powf(c1 < 0.0f ? 0.0f : c1, invGamma);
    c2 = powf(c2 < 0.0f ? 0.0f : c2, invGamma);
}

// Narkowicz 2015 ACES RRT+ODT approximation
__host__ __device__ static inline float hdr_acesFilmic(float x)
{
    return hdr_clamp01((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f));
}

// Reinhard et al. 2002 extended operator with white-point
__global__ static void k_hdr_reinhardGlobal(const unsigned char *__restrict__ in,
                                              unsigned char *__restrict__ out,
                                              int W, int H,
                                              float exposure, float white2,
                                              float sat, float invGamma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i = (y * W + x) * 3;
    float c0 = in[i+0] / 255.0f * exposure;
    float c1 = in[i+1] / 255.0f * exposure;
    float c2 = in[i+2] / 255.0f * exposure;

    float L     = hdr_bgr2lum(c0, c1, c2);
    float L_out = L * (1.0f + L / white2) / (1.0f + L);
    float scale = (L > 1e-6f) ? (L_out / L) : 0.0f;
    c0 *= scale; c1 *= scale; c2 *= scale;

    hdr_satGamma(c0, c1, c2, sat, invGamma);
    out[i+0] = hdr_toUchar(c0);
    out[i+1] = hdr_toUchar(c1);
    out[i+2] = hdr_toUchar(c2);
}

__global__ static void k_hdr_aces(const unsigned char *__restrict__ in,
                                   unsigned char *__restrict__ out,
                                   int W, int H,
                                   float exposure, float sat, float invGamma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i = (y * W + x) * 3;
    float c0 = hdr_acesFilmic(in[i+0] / 255.0f * exposure);
    float c1 = hdr_acesFilmic(in[i+1] / 255.0f * exposure);
    float c2 = hdr_acesFilmic(in[i+2] / 255.0f * exposure);

    hdr_satGamma(c0, c1, c2, sat, invGamma);
    out[i+0] = hdr_toUchar(c0);
    out[i+1] = hdr_toUchar(c1);
    out[i+2] = hdr_toUchar(c2);
}

__global__ static void k_hdr_toFloat(const unsigned char *__restrict__ in,
                                      float *__restrict__ out,
                                      int W, int H, float exposure)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i = (y * W + x) * 3;
    out[i+0] = in[i+0] / 255.0f * exposure;
    out[i+1] = in[i+1] / 255.0f * exposure;
    out[i+2] = in[i+2] / 255.0f * exposure;
}

__global__ static void k_hdr_localReinhard(const float *__restrict__ in,
                                            unsigned char *__restrict__ out,
                                            int W, int H,
                                            float sat, float invGamma)
{
    __shared__ float shLum[HDR_SH_H][HDR_SH_W];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * HDR_TILE_W;
    int by = blockIdx.y * HDR_TILE_H;

    for (int dy = ty; dy < HDR_SH_H; dy += HDR_TILE_H)
        for (int dx = tx; dx < HDR_SH_W; dx += HDR_TILE_W)
        {
            int gx = min(max(bx + dx - HDR_HALO, 0), W - 1);
            int gy = min(max(by + dy - HDR_HALO, 0), H - 1);
            int gi = (gy * W + gx) * 3;
            shLum[dy][dx] = hdr_bgr2lum(in[gi+0], in[gi+1], in[gi+2]);
        }
    __syncthreads();

    int x = bx + tx, y = by + ty;
    if (x >= W || y >= H) return;

    float sumL = 0.0f;
    #pragma unroll
    for (int dy = 0; dy < 2 * HDR_HALO + 1; ++dy)
        #pragma unroll
        for (int dx = 0; dx < 2 * HDR_HALO + 1; ++dx)
            sumL += shLum[ty + dy][tx + dx];

    float L_local = sumL / ((float)((2 * HDR_HALO + 1) * (2 * HDR_HALO + 1)));

    int gi = (y * W + x) * 3;
    float c0 = in[gi+0], c1 = in[gi+1], c2 = in[gi+2];
    float scale = 1.0f / (1.0f + L_local);
    c0 *= scale; c1 *= scale; c2 *= scale;

    hdr_satGamma(c0, c1, c2, sat, invGamma);
    out[gi+0] = hdr_toUchar(c0);
    out[gi+1] = hdr_toUchar(c1);
    out[gi+2] = hdr_toUchar(c2);
}

void applyHDRFilterGPU(const cv::Mat &input, cv::Mat &output, const HdrOptions &opts)
{
    if (input.empty() || input.channels() != 3)
    {
        PLOG_ERROR << "HDR filter requires a non-empty 3-channel image";
        return;
    }

    output.create(input.size(), input.type());

    int W = input.cols, H = input.rows;
    size_t imgBytes   = (size_t)W * H * 3;
    size_t floatBytes = imgBytes * sizeof(float);

    unsigned char *d_in = nullptr, *d_out = nullptr;
    float         *d_float = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_in,  imgBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, imgBytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, input.data, imgBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(cuda::divUp(W, 16), cuda::divUp(H, 16));

    float invGamma = 1.0f / opts.gamma;
    float white2   = opts.whitePoint * opts.whitePoint;

    if (opts.algorithm == "aces")
    {
        k_hdr_aces<<<grid, block>>>(d_in, d_out, W, H,
                                    opts.exposure, opts.saturation, invGamma);
    }
    else if (opts.algorithm == "local")
    {
        CHECK_CUDA_ERROR(cudaMalloc(&d_float, floatBytes));
        k_hdr_toFloat<<<grid, block>>>(d_in, d_float, W, H, opts.exposure);
        k_hdr_localReinhard<<<grid, block>>>(d_float, d_out, W, H,
                                             opts.saturation, invGamma);
        cudaFree(d_float);
    }
    else
    {
        k_hdr_reinhardGlobal<<<grid, block>>>(d_in, d_out, W, H,
                                              opts.exposure, white2,
                                              opts.saturation, invGamma);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_out, imgBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
}

void applyHDRFilterCPU(const cv::Mat &input, cv::Mat &output, const HdrOptions &opts)
{
    if (input.empty() || input.channels() != 3)
    {
        PLOG_ERROR << "HDR filter requires a non-empty 3-channel image";
        return;
    }

    output.create(input.size(), input.type());

    int W = input.cols, H = input.rows;
    float invGamma = 1.0f / opts.gamma;
    float white2   = opts.whitePoint * opts.whitePoint;

    const unsigned char *in  = input.data;
    unsigned char       *out = output.data;

    if (opts.algorithm == "aces")
    {
        for (int y = 0; y < H; ++y)
        for (int x = 0; x < W;  ++x)
        {
            int i = (y * W + x) * 3;
            float c0 = hdr_acesFilmic(in[i+0] / 255.0f * opts.exposure);
            float c1 = hdr_acesFilmic(in[i+1] / 255.0f * opts.exposure);
            float c2 = hdr_acesFilmic(in[i+2] / 255.0f * opts.exposure);
            hdr_satGamma(c0, c1, c2, opts.saturation, invGamma);
            out[i+0] = hdr_toUchar(c0);
            out[i+1] = hdr_toUchar(c1);
            out[i+2] = hdr_toUchar(c2);
        }
    }
    else if (opts.algorithm == "local")
    {
        std::vector<float> fbuf((size_t)W * H * 3);
        for (int j = 0; j < W * H * 3; ++j)
            fbuf[j] = in[j] / 255.0f * opts.exposure;

        constexpr int WIN = (2 * HDR_HALO + 1) * (2 * HDR_HALO + 1);

        for (int y = 0; y < H; ++y)
        for (int x = 0; x < W;  ++x)
        {
            float sumL = 0.0f;
            for (int dy = -HDR_HALO; dy <= HDR_HALO; ++dy)
            for (int dx = -HDR_HALO; dx <= HDR_HALO; ++dx)
            {
                int ix = std::min(std::max(x + dx, 0), W - 1);
                int iy = std::min(std::max(y + dy, 0), H - 1);
                int ii = (iy * W + ix) * 3;
                sumL += hdr_bgr2lum(fbuf[ii+0], fbuf[ii+1], fbuf[ii+2]);
            }
            float L_local = sumL / WIN;
            int i = (y * W + x) * 3;
            float c0 = fbuf[i+0], c1 = fbuf[i+1], c2 = fbuf[i+2];
            float scale = 1.0f / (1.0f + L_local);
            c0 *= scale; c1 *= scale; c2 *= scale;
            hdr_satGamma(c0, c1, c2, opts.saturation, invGamma);
            out[i+0] = hdr_toUchar(c0);
            out[i+1] = hdr_toUchar(c1);
            out[i+2] = hdr_toUchar(c2);
        }
    }
    else
    {
        for (int y = 0; y < H; ++y)
        for (int x = 0; x < W;  ++x)
        {
            int i = (y * W + x) * 3;
            float c0 = in[i+0] / 255.0f * opts.exposure;
            float c1 = in[i+1] / 255.0f * opts.exposure;
            float c2 = in[i+2] / 255.0f * opts.exposure;
            float L     = hdr_bgr2lum(c0, c1, c2);
            float L_out = L * (1.0f + L / white2) / (1.0f + L);
            float scale = (L > 1e-6f) ? (L_out / L) : 0.0f;
            c0 *= scale; c1 *= scale; c2 *= scale;
            hdr_satGamma(c0, c1, c2, opts.saturation, invGamma);
            out[i+0] = hdr_toUchar(c0);
            out[i+1] = hdr_toUchar(c1);
            out[i+2] = hdr_toUchar(c2);
        }
    }
}

} // namespace cuda_filter
