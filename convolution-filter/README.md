# Image Convolution Filter — CUDA Assignment

Three-part implementation of 2D image convolution in CUDA:

- **Part 1** — CPU baseline + naive GPU kernel (filter in constant memory)
- **Part 2** — Shared-memory tiled kernel + separable two-pass Gaussian
- **Part 3** — Automated benchmarks across image sizes and filter types

Benchmarks run on an **NVIDIA GeForce RTX 3050 Ti Laptop GPU** (CUDA 13.2, Nsight Systems 2025.6.3).

## Build & Run

```cmd
nvcc -O2 -o convolution_filter.exe convolution_filter.cu
.\convolution_filter.exe
```

Run inside the **x64 Native Tools Command Prompt for VS 2022** so `nvcc` is on `PATH`.

---

## System Specs

**Host**

| Property | Value |
|----------|-------|
| OS | Microsoft Windows 11 Home (Build 10.0.26200) |
| CPU | 11th Gen Intel Core i5-11300H @ 3.10 GHz |
| System RAM | 32 GB |

**GPU**

| Property | Value |
|----------|-------|
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| Architecture | Ampere (Compute Capability 8.6) |
| Streaming Multiprocessors | 20 SMs |
| VRAM | 4096 MiB (4.0 GB) |
| Max SM Clock | 2100 MHz |
| Max Memory Clock | 6001 MHz |
| Memory Bus Width | 128-bit |
| Theoretical Memory Bandwidth | ~192 GB/s |
| PCIe Interface | Gen 3 × 16 |
| NVIDIA Driver | 595.71 |
| CUDA Version | 13.2 |
| Nsight Systems | 2025.6.3 |

---

## What is Image Convolution?

A convolution filter replaces each pixel with a weighted sum of its neighbourhood. For a pixel at `(x, y)` and a filter of half-radius `r`:

```
out[y][x] = Σ_{fy=0..fw-1} Σ_{fx=0..fw-1}  filter[fy][fx] × in[y+fy-r][x+fx-r]
```

Out-of-bounds pixels are handled with **clamp-to-border** (edge pixels are repeated). The result is clamped to `[0, 255]` before being stored as `unsigned char`.

---

## Filters

| Name | Size | Purpose |
|------|------|---------|
| `kBoxBlur3` | 3×3 | Uniform average over a 3×3 neighbourhood |
| `kGaussian5` | 5×5 | Weighted Gaussian blur (sum = 1, weights from Pascal's triangle) |
| `kSharpen3` | 3×3 | Enhances edges by subtracting blurred neighbours |
| `kSobelX3` | 3×3 | Detects horizontal edges (gradient in X direction) |
| `kGaussian5Sep` | 1×5 | 1D Gaussian used for the separable two-pass path |

---

## Constants

```c
const int REPS = 5;   // kernel launches averaged for GPU timing
```

Block shapes used:

| Config | Block dim | Threads/block | Grid |
|--------|-----------|---------------|------|
| Naive / separable | `(16, 16)` | 256 | `ceil(W/16) × ceil(H/16)` |
| Shared tile=16 | `(16, 16)` | 256 | `ceil(W/16) × ceil(H/16)` |
| Shared tile=32 | `(32, 32)` | 1024 | `ceil(W/32) × ceil(H/32)` |

The tile=32 block hits the 1024-thread-per-block maximum; tile=16 leaves headroom for the scheduler to run multiple blocks per SM.

---

## Implementation Breakdown

### 1. CPU Baseline — `convolutionCPU`

```c
void convolutionCPU(const Image *in, Image *out, const float *filter, int fw) {
    int r = fw / 2;
    for (int y = 0; y < in->height; y++)
        for (int x = 0; x < in->width; x++)
            for (int c = 0; c < in->channels; c++) {
                float acc = 0.f;
                for (int fy = 0; fy < fw; fy++)
                    for (int fx = 0; fx < fw; fx++) {
                        /* clamp indices to border */
                        acc += filter[fy * fw + fx] * in->data[...];
                    }
                out->data[...] = (unsigned char)clamp(acc, 0, 255);
            }
}
```

Four nested loops (y → x → channel → filter). Reads from `in` are roughly sequential for small filters, but the inner loop still scatters across rows. Used as the correctness reference.

---

### 2. CPU Separable Baseline — `convolutionCPU_Sep`

For a separable filter `f2D = f1D ⊗ f1D`, the 2D convolution decomposes into two 1D passes:

1. **Horizontal pass** — convolve each row with `f1D`, write `float` intermediates to `tmp`.
2. **Vertical pass** — convolve each column of `tmp` with `f1D`, write `unsigned char` to `out`.

Cost drops from `O(fw²)` multiply-adds per pixel to `O(2 × fw)`. Used as the reference for the GPU separable path.

---

### 3. GPU Constant Memory — `convolutionNaive`

```cuda
__constant__ float d_filter[81];   // max 9×9 filter, broadcast-cached

__global__ void convolutionNaive(...) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    /* one thread per output pixel; inner loops read d_filter + global in[] */
}
```

**Constant memory** (`__constant__`) is cached in a dedicated per-SM L1 constant cache. When all threads in a warp read the same filter element (uniform access), it delivers a single broadcast at ~shared-memory latency. The bottleneck here is the repeated global-memory reads of `in[]` — each output pixel re-reads its entire `fw × fw` neighbourhood from L2/DRAM.

---

### 4. GPU Shared Memory — `convolutionShared<BLOCK>`

```cuda
template<int BLOCK>
__global__ void convolutionShared(...) {
    extern __shared__ float smem[];   // (BLOCK + 2r)² × channels floats
    int smDim = BLOCK + 2 * r;

    // Cooperatively load halo tile into shared memory
    for (int dy = ty; dy < smDim; dy += BLOCK)
        for (int dx = tx; dx < smDim; dx += BLOCK)
            tile[dy * smDim + dx] = in[clamped global address];
    __syncthreads();

    // Each thread computes its output pixel from shared memory only
    for (int fy = 0; fy < fw; fy++)
        for (int fx = 0; fx < fw; fx++)
            acc += d_filter[fy * fw + fx] * tile[(ty + fy) * smDim + (tx + fx)];
}
```

Each block loads a `(BLOCK + 2r) × (BLOCK + 2r)` **halo tile** into shared memory once, then all threads read their filter taps from shared memory instead of global. This transforms `BLOCK²` independent global-memory streams into one cooperative load, reducing global traffic by up to `fw²` times for large tiles.

The `__syncthreads()` between channels prevents the next channel's cooperative load from racing the current channel's compute.

---

### 5. GPU Separable Two-Pass — `sepH` + `sepV`

```cuda
__constant__ float d_filterSep[9];  // 1D filter

__global__ void sepH(const unsigned char *in, float *tmp, ...) {
    /* convolve horizontally; store float result in tmp */
}
__global__ void sepV(const float *tmp, unsigned char *out, ...) {
    /* convolve vertically; clamp and store uchar */
}
```

Two kernel launches replace the single `fw × fw` inner loop with two `fw`-tap passes. For a 5×5 Gaussian this cuts arithmetic from 25 multiply-adds to 10 per pixel. The intermediate buffer `tmp` is stored as `float` to avoid rounding accumulation between passes.

---

### 6. Error Checking — `CHECK_CUDA`

```c
#define CHECK_CUDA(call) \
    do { cudaError_t _e = (call); \
         if (_e != cudaSuccess) { fprintf(stderr, ...); exit(EXIT_FAILURE); } \
    } while (0)
```

Every CUDA API call is wrapped so a failed allocation or copy immediately prints the file and line number and aborts rather than silently producing wrong output.

---

### 7. Timing — CUDA Events

```c
cudaEvent_t ev_s, ev_e;
cudaEventRecord(ev_s);
for (int i = 0; i < REPS; i++)
    kernel<<<grid, block, smem>>>(d_in, d_out, ...);
cudaEventRecord(ev_e);
cudaEventSynchronize(ev_e);
float ms = elapsedMs(ev_s, ev_e) / REPS;
```

CUDA events are stamped on the GPU timeline, giving sub-microsecond accuracy regardless of host scheduling latency. The average of `REPS=5` runs smooths out one-off launch overhead.

---

### 8. Verification — `verify`

```c
bool verify(const unsigned char *ref, const unsigned char *res, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (abs((int)ref[i] - (int)res[i]) > 1) return false;
    return true;
}
```

Allows ±1 tolerance to account for the different float→`unsigned char` rounding between scalar CPU and parallel GPU (different FMA order). The `[OK]` / `[FAIL]` column reflects this check.

---

## Results (RTX 3050 Ti Laptop, Ampere 8.6, 20 SMs)

### BoxBlur 3×3 — CPU vs GPU

| Image size | CPU (ms) | GPU naive (ms) | GPU shared tile=16 (ms) | GPU shared tile=32 (ms) |
|------------|----------|----------------|------------------------|------------------------|
| 512×512 | 7.000 | 0.025 | 0.021 | 0.032 |
| 1024×1024 | 26.000 | 0.085 | 0.088 | 0.112 |
| 2048×2048 | 101.000 | 0.314 | **0.286** | 0.484 |
| 4096×4096 | 415.000 | 1.243 | **1.110** | 1.932 |

CPU → GPU speedup at 4096×4096: **333× (naive)**, **374× (shared tile=16)**.

---

### Gaussian 5×5 — including separable two-pass

| Image size | CPU 2D (ms) | CPU sep (ms) | GPU naive (ms) | GPU tile=16 (ms) | GPU tile=32 (ms) | GPU sep (ms) |
|------------|------------|--------------|----------------|-----------------|-----------------|--------------|
| 512×512 | 12.000 | 5.000 | 0.049 | **0.035** | 0.047 | 0.039 |
| 1024×1024 | 46.000 | 18.000 | 0.173 | **0.120** | 0.171 | 0.096 |
| 2048×2048 | 186.000 | 72.000 | 0.676 | 0.468 | 0.721 | **0.357** |
| 4096×4096 | 735.000 | 300.000 | 2.698 | 1.848 | 2.877 | **1.386** |

CPU → GPU speedup at 4096×4096: **273× (naive)**, **398× (shared tile=16)**, **530× (separable)**.

---

### Additional Filters — 2048×2048

| Filter | CPU (ms) | GPU naive (ms) | GPU tile=16 (ms) | GPU tile=32 (ms) |
|--------|----------|----------------|-----------------|-----------------|
| Sharpen 3×3 | 100.000 | 0.315 | **0.285** | 0.484 |
| SobelX 3×3 | 106.000 | 0.314 | **0.285** | 0.484 |

