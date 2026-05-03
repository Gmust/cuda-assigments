// Matrix Transposition Assignment
// Part 1: CPU baseline + naive GPU kernel
// Part 2: Shared-memory tiled kernel + unified memory + block-size analysis
// Part 3: Performance numbers printed automatically


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Error checking
inline cudaError_t checkCuda(cudaError_t r) {
    if (r != cudaSuccess)
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
    return r;
}

// Part 1 — CPU baseline
// in  : rows x cols, row-major
// out : cols x rows, row-major (transposed)
void transposeCPU(const float *in, float *out, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            out[(size_t)c * rows + r] = in[(size_t)r * cols + c];
}

// Part 1 — Naive GPU kernel
// Each thread writes one output element.
// Reads are coalesced (consecutive cols in a warp), but
// writes stride by `rows` → non-coalesced.
__global__ void transposeNaive(float *out, const float *in, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows)
        out[(size_t)col * rows + row] = in[(size_t)row * cols + col];
}

// Part 2 — Shared-memory tiled kernel
// Template params: TILE = tile width, BROWS = threads per block in y.
// Each thread loads TILE/BROWS elements per launch, giving
// coalesced reads AND writes via the transposed tile in SMEM.
// Padding (+1) eliminates shared-mem bank conflicts on the
// column-strided access in the write phase.
template <int TILE, int BROWS>
__global__ void transposeSharedMem(float *out, const float *in, int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1]; // +1 avoids bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y; 
    for (int j = 0; j < TILE; j += BROWS) {
        if (x < cols && (y + j) < rows)
            tile[threadIdx.y + j][threadIdx.x] = in[(size_t)(y + j) * cols + x];
    }

    __syncthreads();

    // Write phase: threads write consecutive output columns → coalesced
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y; 
    for (int j = 0; j < TILE; j += BROWS) {
        if (x < rows && (y + j) < cols)
            out[(size_t)(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// Correctness check — compare against CPU reference
bool verify(const float *ref, const float *result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (fabsf(ref[i] - result[i]) > 1e-5f) {
            fprintf(stderr, "  Mismatch at [%zu]: ref=%.6f  got=%.6f\n",
                    i, ref[i], result[i]);
            return false;
        }
    }
    return true;
}

// Timing / display helpers
static float elapsedMs(cudaEvent_t s, cudaEvent_t e) {
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    return ms;
}

static void printRow(const char *label, int rows, int cols, float ms, bool ok) {
    double bytes = 2.0 * rows * cols * sizeof(float);
    double gbs   = bytes / (1e9 * ms / 1000.0);
    printf("  %-46s  %8.3f ms  %6.2f GB/s  [%s]\n",
           label, ms, gbs, ok ? "OK" : "FAIL");
}

// Run one tiled configuration on device memory
template <int TILE, int BROWS>
static void runTiledConfig(const char *label,
                           float *d_out, const float *d_in,
                           float *h_out, const float *h_ref,
                           int rows, int cols, int reps,
                           cudaEvent_t ev_s, cudaEvent_t ev_e)
{
    size_t N = (size_t)rows * cols;
    dim3 block(TILE, BROWS);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

    transposeSharedMem<TILE, BROWS><<<grid, block>>>(d_out, d_in, rows, cols);
    cudaDeviceSynchronize(); 

    cudaEventRecord(ev_s);
    for (int i = 0; i < reps; i++)
        transposeSharedMem<TILE, BROWS><<<grid, block>>>(d_out, d_in, rows, cols);
    cudaEventRecord(ev_e);
    cudaEventSynchronize(ev_e);

    checkCuda(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    printRow(label, rows, cols, elapsedMs(ev_s, ev_e) / reps,
             verify(h_ref, h_out, N));
}

// Benchmark one matrix size — runs all implementations
void benchmark(int rows, int cols, int deviceId, int concurrentManaged,
               cudaEvent_t ev_s, cudaEvent_t ev_e)
{
    printf("\n=== %d x %d matrix ===\n", rows, cols);
    const int REPS = 10;
    size_t N = (size_t)rows * cols;

    float *h_in  = (float *)malloc(N * sizeof(float));
    float *h_ref = (float *)malloc(N * sizeof(float));
    float *h_out = (float *)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
        h_in[i] = (float)(rand() % 10000) / 100.0f;

    clock_t t0 = clock();
    transposeCPU(h_in, h_ref, rows, cols);
    clock_t t1 = clock();
    printf("  %-46s  %8.3f ms\n", "CPU",
           1000.0f * (float)(t1 - t0) / CLOCKS_PER_SEC);

    float *d_in, *d_out;
    checkCuda(cudaMalloc(&d_in,  N * sizeof(float)));
    checkCuda(cudaMalloc(&d_out, N * sizeof(float)));
    checkCuda(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // --- Part 1: Naive GPU (32x32 blocks) ---
    {
        dim3 block(32, 32);
        dim3 grid((cols + 31) / 32, (rows + 31) / 32);

        transposeNaive<<<grid, block>>>(d_out, d_in, rows, cols);
        cudaDeviceSynchronize(); 

        cudaEventRecord(ev_s);
        for (int i = 0; i < REPS; i++)
            transposeNaive<<<grid, block>>>(d_out, d_in, rows, cols);
        cudaEventRecord(ev_e);
        cudaEventSynchronize(ev_e);

        checkCuda(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
        printRow("GPU naive (32x32)", rows, cols,
                 elapsedMs(ev_s, ev_e) / REPS, verify(h_ref, h_out, N));
    }

    // --- Part 2: Shared-mem tiled — block-size analysis ---
    // tile=8,  block=8x8  (64  threads, 1 element/thread)
    runTiledConfig< 8,  8>("GPU tiled 8x8   (tile=8,  block=8x8 )",
                           d_out, d_in, h_out, h_ref, rows, cols, REPS, ev_s, ev_e);
    // tile=16, block=16x16 (256 threads, 1 element/thread)
    runTiledConfig<16, 16>("GPU tiled 16x16 (tile=16, block=16x16)",
                           d_out, d_in, h_out, h_ref, rows, cols, REPS, ev_s, ev_e);
    // tile=32, block=32x8  (256 threads, 4 elements/thread) — typical sweet spot
    runTiledConfig<32,  8>("GPU tiled 32x8  (tile=32, block=32x8 )",
                           d_out, d_in, h_out, h_ref, rows, cols, REPS, ev_s, ev_e);
    // tile=32, block=32x32 (1024 threads, 1 element/thread)
    runTiledConfig<32, 32>("GPU tiled 32x32 (tile=32, block=32x32)",
                           d_out, d_in, h_out, h_ref, rows, cols, REPS, ev_s, ev_e);

    // --- Part 2: Unified memory + prefetch (best tiled config: tile=32, 32x8 block) ---
    {
        float *um_in, *um_out;
        checkCuda(cudaMallocManaged(&um_in,  N * sizeof(float)));
        checkCuda(cudaMallocManaged(&um_out, N * sizeof(float)));
        for (size_t i = 0; i < N; i++) um_in[i] = h_in[i];

        if (concurrentManaged) {
            cudaMemLocation gpuLoc = {cudaMemLocationTypeDevice, deviceId};
            checkCuda(cudaMemPrefetchAsync(um_in,  N * sizeof(float), gpuLoc, 0, 0));
            checkCuda(cudaMemPrefetchAsync(um_out, N * sizeof(float), gpuLoc, 0, 0));
            cudaDeviceSynchronize();
        }

        dim3 block(32, 8);
        dim3 grid((cols + 31) / 32, (rows + 31) / 32);

        transposeSharedMem<32, 8><<<grid, block>>>(um_out, um_in, rows, cols);
        cudaDeviceSynchronize(); // warm-up

        cudaEventRecord(ev_s);
        for (int i = 0; i < REPS; i++)
            transposeSharedMem<32, 8><<<grid, block>>>(um_out, um_in, rows, cols);
        cudaEventRecord(ev_e);
        cudaEventSynchronize(ev_e);

        if (concurrentManaged) {
            cudaMemLocation cpuLoc = {cudaMemLocationTypeHost, 0};
            checkCuda(cudaMemPrefetchAsync(um_out, N * sizeof(float), cpuLoc, 0, 0));
            cudaDeviceSynchronize();
        }

        const char *suffix = concurrentManaged
            ? "GPU tiled 32x8 + unified+prefetch"
            : "GPU tiled 32x8 + unified (no prefetch/WDDM)";
        printRow(suffix, rows, cols, elapsedMs(ev_s, ev_e) / REPS,
                 verify(h_ref, um_out, N));

        checkCuda(cudaFree(um_in));
        checkCuda(cudaFree(um_out));
    }

    checkCuda(cudaFree(d_in));
    checkCuda(cudaFree(d_out));
    free(h_in);
    free(h_ref);
    free(h_out);
}

// Main
int main() {
    srand(42);

    int deviceId;
    checkCuda(cudaGetDevice(&deviceId));
    cudaDeviceProp props;
    checkCuda(cudaGetDeviceProperties(&props, deviceId));

    int concurrentManaged = 0;
    cudaDeviceGetAttribute(&concurrentManaged,
                           cudaDevAttrConcurrentManagedAccess, deviceId);

    printf("GPU: %s  (SMs: %d, mem: %.1f GB, concurrent managed: %s)\n\n",
           props.name,
           props.multiProcessorCount,
           (double)props.totalGlobalMem / 1e9,
           concurrentManaged ? "yes" : "no (WDDM)");

    printf("label                                               avg time    bandwidth  ok?\n");
    printf("-------------------------------------------------------------------------------\n");
    printf("(averages of 10 launches; bandwidth = 2 * rows * cols * 4 bytes / time)\n");

    cudaEvent_t ev_s, ev_e;
    checkCuda(cudaEventCreate(&ev_s));
    checkCuda(cudaEventCreate(&ev_e));

    // Start at 2048x1024 and scale up as required by the assignment
    benchmark( 2048,  1024, deviceId, concurrentManaged, ev_s, ev_e);
    benchmark( 4096,  2048, deviceId, concurrentManaged, ev_s, ev_e);
    benchmark( 8192,  4096, deviceId, concurrentManaged, ev_s, ev_e);
    benchmark(16384,  8192, deviceId, concurrentManaged, ev_s, ev_e);

    checkCuda(cudaEventDestroy(ev_s));
    checkCuda(cudaEventDestroy(ev_e));
    return 0;
}
