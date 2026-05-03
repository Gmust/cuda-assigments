#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

const int TILE_DIM  = 32;
const int BLOCK_ROWS = 8;  
const int NUM_REPS  = 100; 

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess)
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(result));
    return result;
}

void transposeCPU(const float *in, float *out, int n) {
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            out[c * n + r] = in[r * n + c];
}

void postprocess(const char *label, const float *ref, const float *res,
                 int n, float ms) {
    for (int i = 0; i < n * n; i++) {
        if (res[i] != ref[i]) {
            printf("  %-30s  *** FAILED at [%d]: ref=%.1f got=%.1f\n",
                   label, i, ref[i], res[i]);
            return;
        }
    }
    double gb   = 2.0 * n * n * sizeof(float) * NUM_REPS * 1e-9;
    double secs = ms * 1e-3;
    printf("  %-30s  %8.2f GB/s  %8.2f ms/rep\n",
           label, gb / secs, ms / NUM_REPS);
}

__global__ void copy(float *out, const float *in, int n) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        out[(y + j) * n + x] = in[(y + j) * n + x];
}

__global__ void transposeNaive(float *out, const float *in, int n) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        out[x * n + (y + j)] = in[(y + j) * n + x];
}

__global__ void transposeCoalesced(float *out, const float *in, int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void transposeNoBankConflicts(float *out, const float *in, int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
}

void benchmark(int n) {
    printf("\n=== %d x %d matrix ===\n", n, n);
    size_t bytes = (size_t)n * n * sizeof(float);

    float *h_in  = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);

    for (int i = 0; i < n * n; i++) h_in[i] = (float)i;

    clock_t t0 = clock();
    transposeCPU(h_in, h_ref, n);
    clock_t t1 = clock();
    printf("  %-30s  %8.2f ms\n", "CPU",
           1000.0 * (double)(t1 - t0) / CLOCKS_PER_SEC);

    float *d_in, *d_out;
    checkCuda(cudaMalloc(&d_in,  bytes));
    checkCuda(cudaMalloc(&d_out, bytes));
    checkCuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 grid(n / TILE_DIM, n / TILE_DIM);
    dim3 block(TILE_DIM, BLOCK_ROWS);

    cudaEvent_t ev_s, ev_e;
    checkCuda(cudaEventCreate(&ev_s));
    checkCuda(cudaEventCreate(&ev_e));
    float ms;

#define RUN(kernel, ref_data)                                             
    kernel<<<grid, block>>>(d_out, d_in, n);          
    checkCuda(cudaEventRecord(ev_s));                                     
    for (int i = 0; i < NUM_REPS; i++)                                    
        kernel<<<grid, block>>>(d_out, d_in, n);                          
    checkCuda(cudaEventRecord(ev_e));                                     
    checkCuda(cudaEventSynchronize(ev_e));                                
    checkCuda(cudaEventElapsedTime(&ms, ev_s, ev_e));                     
    checkCuda(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));   
    postprocess(#kernel, ref_data, h_out, n, ms)

    RUN(copy,                    h_in);
    RUN(transposeNaive,          h_ref);
    RUN(transposeCoalesced,      h_ref);
    RUN(transposeNoBankConflicts, h_ref);

#undef RUN

    checkCuda(cudaEventDestroy(ev_s));
    checkCuda(cudaEventDestroy(ev_e));
    checkCuda(cudaFree(d_in));
    checkCuda(cudaFree(d_out));
    free(h_in); free(h_ref); free(h_out);
}

int main(void) {
    int devId;
    checkCuda(cudaGetDevice(&devId));
    cudaDeviceProp props;
    checkCuda(cudaGetDeviceProperties(&props, devId));
    printf("GPU: %s\n", props.name);
    printf("TILE_DIM=%d  BLOCK_ROWS=%d  NUM_REPS=%d\n\n",
           TILE_DIM, BLOCK_ROWS, NUM_REPS);
    printf("  %-30s  %10s  %12s\n", "Kernel", "Bandwidth", "ms/rep");
    printf("  %s\n", "--------------------------------------------------------------");

    benchmark(1024);
    benchmark(2048);
    benchmark(4096);

    return 0;
}
