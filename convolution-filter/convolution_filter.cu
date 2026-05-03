// Image Convolution Assignment
// Part 1: CPU baseline + naive GPU kernel (constant memory for filter)
// Part 2: Shared-memory tiled kernel + separable Gaussian two-pass
// Part 3: Performance numbers printed automatically across image/filter sizes

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

static const float kBoxBlur3[9] = {
    1/9.f, 1/9.f, 1/9.f,
    1/9.f, 1/9.f, 1/9.f,
    1/9.f, 1/9.f, 1/9.f
};

static const float kGaussian5[25] = {
    1/273.f,  4/273.f,  7/273.f,  4/273.f, 1/273.f,
    4/273.f, 16/273.f, 26/273.f, 16/273.f, 4/273.f,
    7/273.f, 26/273.f, 41/273.f, 26/273.f, 7/273.f,
    4/273.f, 16/273.f, 26/273.f, 16/273.f, 4/273.f,
    1/273.f,  4/273.f,  7/273.f,  4/273.f, 1/273.f
};

static const float kSharpen3[9] = {
     0.f, -1.f,  0.f,
    -1.f,  5.f, -1.f,
     0.f, -1.f,  0.f
};

static const float kSobelX3[9] = {
    -1.f, 0.f, 1.f,
    -2.f, 0.f, 2.f,
    -1.f, 0.f, 1.f
};

static const float kGaussian5Sep[5] = {
    1/17.f, 4/17.f, 7/17.f, 4/17.f, 1/17.f
};

#define CHECK_CUDA(call)                                                     
    do {                                                                     
        cudaError_t _e = (call);                                             
        if (_e != cudaSuccess) {                                             
            fprintf(stderr, "CUDA Error: %s  at %s:%d\n",                    
                    cudaGetErrorString(_e), __FILE__, __LINE__);             
            exit(EXIT_FAILURE);                                              
        }                                                                    
    } while (0)

typedef struct { unsigned char *data; int width, height, channels; } Image;

static Image allocImage(int w, int h, int c) {
    Image img = { (unsigned char *)malloc((size_t)w * h * c), w, h, c };
    return img;
}
static void freeImage(Image *img) { free(img->data); img->data = NULL; }

static void generateImage(Image *img) {
    for (int y = 0; y < img->height; y++)
        for (int x = 0; x < img->width; x++)
            for (int c = 0; c < img->channels; c++)
                img->data[(y * img->width + x) * img->channels + c] =
                    (unsigned char)((x * 3 + y * 7 + c * 50) & 0xFF);
}


void convolutionCPU(const Image *in, Image *out,
                    const float *filter, int fw) {
    int r = fw / 2;
    for (int y = 0; y < in->height; y++) {
        for (int x = 0; x < in->width; x++) {
            for (int c = 0; c < in->channels; c++) {
                float acc = 0.f;
                for (int fy = 0; fy < fw; fy++) {
                    for (int fx = 0; fx < fw; fx++) {
                        int iy = y + fy - r;
                        int ix = x + fx - r;
                        iy = (iy < 0) ? 0 : (iy >= in->height ? in->height-1 : iy);
                        ix = (ix < 0) ? 0 : (ix >= in->width  ? in->width -1 : ix);
                        acc += filter[fy * fw + fx] *
                               in->data[(iy * in->width + ix) * in->channels + c];
                    }
                }
                acc = fminf(fmaxf(acc, 0.f), 255.f);
                out->data[(y * out->width + x) * out->channels + c] = (unsigned char)acc;
            }
        }
    }
}

static void convolutionCPU_Sep(const Image *in, Image *out,
                                const float *f1D, int fw) {
    int W = in->width, H = in->height, C = in->channels;
    int r = fw / 2;
    float *tmp = (float *)malloc((size_t)W * H * C * sizeof(float));

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < C; c++) {
                float acc = 0.f;
                for (int fx = 0; fx < fw; fx++) {
                    int ix = x + fx - r;
                    ix = (ix < 0) ? 0 : (ix >= W ? W-1 : ix);
                    acc += f1D[fx] * in->data[(y * W + ix) * C + c];
                }
                tmp[(y * W + x) * C + c] = acc;
            }

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < C; c++) {
                float acc = 0.f;
                for (int fy = 0; fy < fw; fy++) {
                    int iy = y + fy - r;
                    iy = (iy < 0) ? 0 : (iy >= H ? H-1 : iy);
                    acc += f1D[fy] * tmp[(iy * W + x) * C + c];
                }
                acc = fminf(fmaxf(acc, 0.f), 255.f);
                out->data[(y * W + x) * C + c] = (unsigned char)acc;
            }

    free(tmp);
}


__constant__ float d_filter[81];    
__constant__ float d_filterSep[9]; 


__global__ void convolutionNaive(const unsigned char *in, unsigned char *out,
                                  int fw, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int r = fw / 2;
    for (int c = 0; c < channels; c++) {
        float acc = 0.f;
        for (int fy = 0; fy < fw; fy++) {
            for (int fx = 0; fx < fw; fx++) {
                int iy = y + fy - r;
                int ix = x + fx - r;
                iy = (iy < 0) ? 0 : (iy >= height ? height-1 : iy);
                ix = (ix < 0) ? 0 : (ix >= width  ? width -1 : ix);
                acc += d_filter[fy * fw + fx] *
                       in[(iy * width + ix) * channels + c];
            }
        }
        acc = fminf(fmaxf(acc, 0.f), 255.f);
        out[(y * width + x) * channels + c] = (unsigned char)acc;
    }
}

template<int BLOCK>
__global__ void convolutionShared(const unsigned char *in, unsigned char *out,
                                   int fw, int width, int height, int channels) {
    extern __shared__ float smem[];  

    int r     = fw / 2;
    int smDim = BLOCK + 2 * r;
    int tx = threadIdx.x, ty = threadIdx.y;
    int outX = blockIdx.x * BLOCK + tx;
    int outY = blockIdx.y * BLOCK + ty;

    for (int c = 0; c < channels; c++) {
        float *tile = smem + c * smDim * smDim;

        for (int dy = ty; dy < smDim; dy += BLOCK) {
            for (int dx = tx; dx < smDim; dx += BLOCK) {
                int gx = blockIdx.x * BLOCK - r + dx;
                int gy = blockIdx.y * BLOCK - r + dy;
                gx = (gx < 0) ? 0 : (gx >= width  ? width -1 : gx);
                gy = (gy < 0) ? 0 : (gy >= height ? height-1 : gy);
                tile[dy * smDim + dx] = in[(gy * width + gx) * channels + c];
            }
        }
        __syncthreads();

        if (outX < width && outY < height) {
            float acc = 0.f;
            for (int fy = 0; fy < fw; fy++)
                for (int fx = 0; fx < fw; fx++)
                    acc += d_filter[fy * fw + fx] *
                           tile[(ty + fy) * smDim + (tx + fx)];
            acc = fminf(fmaxf(acc, 0.f), 255.f);
            out[(outY * width + outX) * channels + c] = (unsigned char)acc;
        }
        __syncthreads(); 
    }
}

__global__ void sepH(const unsigned char *in, float *tmp,
                     int fw, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int r = fw / 2;
    for (int c = 0; c < channels; c++) {
        float acc = 0.f;
        for (int fx = 0; fx < fw; fx++) {
            int ix = x + fx - r;
            ix = (ix < 0) ? 0 : (ix >= width ? width-1 : ix);
            acc += d_filterSep[fx] * in[(y * width + ix) * channels + c];
        }
        tmp[(y * width + x) * channels + c] = acc;
    }
}

__global__ void sepV(const float *tmp, unsigned char *out,
                     int fw, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int r = fw / 2;
    for (int c = 0; c < channels; c++) {
        float acc = 0.f;
        for (int fy = 0; fy < fw; fy++) {
            int iy = y + fy - r;
            iy = (iy < 0) ? 0 : (iy >= height ? height-1 : iy);
            acc += d_filterSep[fy] * tmp[(iy * width + x) * channels + c];
        }
        acc = fminf(fmaxf(acc, 0.f), 255.f);
        out[(y * width + x) * channels + c] = (unsigned char)acc;
    }
}

static float elapsedMs(cudaEvent_t s, cudaEvent_t e) {
    float ms = 0; cudaEventElapsedTime(&ms, s, e); return ms;
}

static bool verify(const unsigned char *ref, const unsigned char *res, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (abs((int)ref[i] - (int)res[i]) > 1) {
            fprintf(stderr, "  Mismatch at [%zu]: ref=%d got=%d\n",
                    i, (int)ref[i], (int)res[i]);
            return false;
        }
    }
    return true;
}

static void printRow(const char *label, float ms, bool ok) {
    printf("  %-52s  %8.3f ms  [%s]\n", label, ms, ok ? "OK" : "FAIL");
}

static void benchmark(int W, int H, int C,
                      const char *filterName,
                      const float *filter2D, int fw,
                      const float *filter1D,  
                      cudaEvent_t ev_s, cudaEvent_t ev_e)
{
    const int REPS = 5;
    size_t N = (size_t)W * H * C;

    printf("\n=== %dx%d image, %d ch, filter=%s (%dx%d) ===\n",
           W, H, C, filterName, fw, fw);

    Image in  = allocImage(W, H, C);
    Image ref = allocImage(W, H, C);
    Image res = allocImage(W, H, C);
    generateImage(&in);

    clock_t t0 = clock();
    convolutionCPU(&in, &ref, filter2D, fw);
    clock_t t1 = clock();
    printf("  %-52s  %8.3f ms\n", "CPU (reference)",
           1000.0 * (double)(t1 - t0) / CLOCKS_PER_SEC);

    unsigned char *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in,  N));
    CHECK_CUDA(cudaMalloc(&d_out, N));
    CHECK_CUDA(cudaMemcpy(d_in, in.data, N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(d_filter, filter2D, fw * fw * sizeof(float)));

    dim3 block16(16, 16), block32(32, 32);
    dim3 grid16((W+15)/16, (H+15)/16);
    dim3 grid32((W+31)/32, (H+31)/32);

    {
        convolutionNaive<<<grid16, block16>>>(d_in, d_out, fw, W, H, C);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(ev_s));
        for (int i = 0; i < REPS; i++)
            convolutionNaive<<<grid16, block16>>>(d_in, d_out, fw, W, H, C);
        CHECK_CUDA(cudaEventRecord(ev_e));
        CHECK_CUDA(cudaEventSynchronize(ev_e));

        CHECK_CUDA(cudaMemcpy(res.data, d_out, N, cudaMemcpyDeviceToHost));
        printRow("GPU naive (16x16 blocks, constant mem)",
                 elapsedMs(ev_s, ev_e) / REPS, verify(ref.data, res.data, N));
    }

    {
        int smDim  = 16 + 2 * (fw / 2);
        size_t smB = (size_t)smDim * smDim * C * sizeof(float);

        convolutionShared<16><<<grid16, block16, smB>>>(d_in, d_out, fw, W, H, C);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(ev_s));
        for (int i = 0; i < REPS; i++)
            convolutionShared<16><<<grid16, block16, smB>>>(d_in, d_out, fw, W, H, C);
        CHECK_CUDA(cudaEventRecord(ev_e));
        CHECK_CUDA(cudaEventSynchronize(ev_e));

        CHECK_CUDA(cudaMemcpy(res.data, d_out, N, cudaMemcpyDeviceToHost));
        printRow("GPU shared mem (tile=16, 256 threads/block)",
                 elapsedMs(ev_s, ev_e) / REPS, verify(ref.data, res.data, N));
    }

    {
        int smDim  = 32 + 2 * (fw / 2);
        size_t smB = (size_t)smDim * smDim * C * sizeof(float);

        convolutionShared<32><<<grid32, block32, smB>>>(d_in, d_out, fw, W, H, C);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(ev_s));
        for (int i = 0; i < REPS; i++)
            convolutionShared<32><<<grid32, block32, smB>>>(d_in, d_out, fw, W, H, C);
        CHECK_CUDA(cudaEventRecord(ev_e));
        CHECK_CUDA(cudaEventSynchronize(ev_e));

        CHECK_CUDA(cudaMemcpy(res.data, d_out, N, cudaMemcpyDeviceToHost));
        printRow("GPU shared mem (tile=32, 1024 threads/block)",
                 elapsedMs(ev_s, ev_e) / REPS, verify(ref.data, res.data, N));
    }

    if (filter1D) {
        CHECK_CUDA(cudaMemcpyToSymbol(d_filterSep, filter1D, fw * sizeof(float)));

        float *d_tmp;
        CHECK_CUDA(cudaMalloc(&d_tmp, N * sizeof(float)));

        Image refSep = allocImage(W, H, C);
        t0 = clock();
        convolutionCPU_Sep(&in, &refSep, filter1D, fw);
        t1 = clock();
        printf("  %-52s  %8.3f ms\n", "CPU separable (H+V, reference)",
               1000.0 * (double)(t1 - t0) / CLOCKS_PER_SEC);

        sepH<<<grid16, block16>>>(d_in, d_tmp, fw, W, H, C);
        sepV<<<grid16, block16>>>(d_tmp, d_out, fw, W, H, C);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(ev_s));
        for (int i = 0; i < REPS; i++) {
            sepH<<<grid16, block16>>>(d_in, d_tmp, fw, W, H, C);
            sepV<<<grid16, block16>>>(d_tmp, d_out, fw, W, H, C);
        }
        CHECK_CUDA(cudaEventRecord(ev_e));
        CHECK_CUDA(cudaEventSynchronize(ev_e));

        CHECK_CUDA(cudaMemcpy(res.data, d_out, N, cudaMemcpyDeviceToHost));
        printRow("GPU separable (H+V passes, 16x16 blocks)",
                 elapsedMs(ev_s, ev_e) / REPS,
                 verify(refSep.data, res.data, N));

        freeImage(&refSep);
        CHECK_CUDA(cudaFree(d_tmp));
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    freeImage(&in);
    freeImage(&ref);
    freeImage(&res);
}

int main(void) {
    int deviceId;
    CHECK_CUDA(cudaGetDevice(&deviceId));
    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDeviceProperties(&props, deviceId));
    printf("GPU: %s  (SMs: %d, global mem: %.1f GB)\n",
           props.name, props.multiProcessorCount,
           (double)props.totalGlobalMem / 1e9);

    cudaEvent_t ev_s, ev_e;
    CHECK_CUDA(cudaEventCreate(&ev_s));
    CHECK_CUDA(cudaEventCreate(&ev_e));

    printf("\nlabel                                                        avg time  ok?\n");
    printf("--------------------------------------------------------------------------\n");
    printf("(CPU times from clock(); GPU times = average of %d launches)\n", 5);

    benchmark( 512,  512, 1, "BoxBlur3",  kBoxBlur3,  3, NULL,          ev_s, ev_e);
    benchmark(1024, 1024, 1, "BoxBlur3",  kBoxBlur3,  3, NULL,          ev_s, ev_e);
    benchmark(2048, 2048, 1, "BoxBlur3",  kBoxBlur3,  3, NULL,          ev_s, ev_e);
    benchmark(4096, 4096, 1, "BoxBlur3",  kBoxBlur3,  3, NULL,          ev_s, ev_e);

    benchmark( 512,  512, 1, "Gaussian5", kGaussian5, 5, kGaussian5Sep, ev_s, ev_e);
    benchmark(1024, 1024, 1, "Gaussian5", kGaussian5, 5, kGaussian5Sep, ev_s, ev_e);
    benchmark(2048, 2048, 1, "Gaussian5", kGaussian5, 5, kGaussian5Sep, ev_s, ev_e);
    benchmark(4096, 4096, 1, "Gaussian5", kGaussian5, 5, kGaussian5Sep, ev_s, ev_e);

    benchmark(2048, 2048, 1, "Sharpen3",  kSharpen3,  3, NULL,          ev_s, ev_e);
    benchmark(2048, 2048, 1, "SobelX3",   kSobelX3,   3, NULL,          ev_s, ev_e);

    CHECK_CUDA(cudaEventDestroy(ev_s));
    CHECK_CUDA(cudaEventDestroy(ev_e));
    return 0;
}
