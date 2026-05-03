# Matrix Transposition — CUDA Assignment

Based on the [NVIDIA developer blog sample](https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu).

Benchmarks four GPU kernels (plus a CPU baseline) on an NVIDIA RTX 3050 Ti Laptop GPU (CUDA 13.2, Nsight Systems 2025.6.3).

## Build & Run

```cmd
nvcc -O2 -o matrix-transpose.exe matrix-transpose.cu
.\matrix-transpose.exe
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

## What is Matrix Transposition?

Transposing a matrix swaps its rows and columns: element at `(row, col)` moves to `(col, row)`. For a row-major flat array:

```
out[col * n + row] = in[row * n + col]
```

There is **no arithmetic** — only memory reads and writes. This makes it a pure memory-bandwidth benchmark. The effective bandwidth formula used in this code:

```
bandwidth (GB/s) = (2 × n × n × 4 bytes × NUM_REPS) / time_seconds
```

The factor of 2 accounts for one full read and one full write of the matrix.

---

## Constants

```c
const int TILE_DIM   = 32;   // tile width = warp size (required for coalescing)
const int BLOCK_ROWS = 8;    // threads per block in y; each thread handles TILE_DIM/BLOCK_ROWS rows
const int NUM_REPS   = 100;  // repetitions averaged for timing
```

The block shape is always `(TILE_DIM=32, BLOCK_ROWS=8)` — 256 threads per block, each loading 4 elements. This avoids the 1024-thread maximum while keeping full 32-wide coalescing on reads and writes.

---

## Implementation Breakdown

### 1. CPU Baseline — `transposeCPU`

```c
void transposeCPU(const float *in, float *out, int n) {
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            out[c * n + r] = in[r * n + c];
}
```

Reads from `in` are sequential (cache-friendly), but writes to `out` stride by `n` — causing cache thrashing on large matrices.

---

### 2. `copy` — Bandwidth Ceiling Reference

```cuda
__global__ void copy(float *out, const float *in, int n) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        out[(y + j) * n + x] = in[(y + j) * n + x];
}
```

Just copies the matrix without transposing. Both reads and writes are perfectly coalesced (consecutive `x` values in the same row). The bandwidth this achieves is the hardware ceiling — no transpose kernel can beat it. Used as a reference to judge how close the optimized kernels get.

---

### 3. `transposeNaive` — Coalesced Reads, Non-Coalesced Writes

```cuda
__global__ void transposeNaive(float *out, const float *in, int n) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        out[x * n + (y + j)] = in[(y + j) * n + x];
}
```

**Read** `in[(y+j) * n + x]`: consecutive threads read consecutive `x` values in the same row → **coalesced**, 1 transaction per warp.

**Write** `out[x * n + (y+j)]`: consecutive threads write to addresses separated by `n` floats → **non-coalesced**, up to 32 transactions per warp.

Result on 4096×4096: ~74 GB/s vs ~179 GB/s ceiling — wastes ~59% of bandwidth on the write side.

---

### 4. `transposeCoalesced` — Shared Memory Staging (with Bank Conflicts)

```cuda
__global__ void transposeCoalesced(float *out, const float *in, int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Read phase: coalesced loads from global → shared memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];

    __syncthreads();

    // Swap block indices to target the transposed output location
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write phase: coalesced stores from shared → global memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
}
```

Shared memory acts as a staging buffer:

1. **Read phase** — each thread loads one element per iteration with consecutive `x` → coalesced global reads.
2. **`__syncthreads()`** — ensures the tile is fully populated before any thread reads it back.
3. **Write phase** — after swapping block indices the output addresses are consecutive → coalesced global writes.

Reading `tile[threadIdx.x][threadIdx.y + j]` in the write phase accesses elements in the same **column** of the tile. Shared memory is divided into 32 banks; elements in the same column of a 32-wide array all fall in the **same bank** → 32-way bank conflict, serialised to 32 sequential accesses.

Result on 4096×4096: ~178 GB/s — nearly at ceiling despite bank conflicts (the conflict penalty is hidden by memory-level parallelism here, which is expected on Ampere GPUs).

---

### 5. `transposeNoBankConflicts` — Optimal (Padding Eliminates Bank Conflicts)

```cuda
__global__ void transposeNoBankConflicts(float *out, const float *in, int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 padding
    // ... identical logic to transposeCoalesced ...
}
```

The only change is `TILE_DIM + 1` in the tile declaration. This shifts each row of the tile to a different starting bank:

```
Without padding:  tile[0][0], tile[1][0], tile[2][0] ... → all bank 0 → conflict
With +1 padding:  tile[0][0] → bank 0, tile[1][0] → bank 1, tile[2][0] → bank 2 ... → no conflict
```

The extra column is never written or read — it only exists to move the row stride from 32 (= number of banks) to 33, making it coprime to the bank count.

Result on 4096×4096: ~178 GB/s — matches the `copy` ceiling.

---

### 6. Error Checking — `checkCuda`

```c
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess)
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(result));
    return result;
}
```

Every CUDA API call is wrapped so failures print a descriptive message rather than silently producing wrong results.

---

### 7. Timing — CUDA Events

```c
cudaEvent_t ev_s, ev_e;
cudaEventCreate(&ev_s);
cudaEventCreate(&ev_e);
cudaEventRecord(ev_s);
for (int i = 0; i < NUM_REPS; i++)
    kernel<<<grid, block>>>(d_out, d_in, n);
cudaEventRecord(ev_e);
cudaEventSynchronize(ev_e);   // block CPU until GPU finishes
float ms;
cudaEventElapsedTime(&ms, ev_s, ev_e);
```

CUDA events are timestamped on the GPU timeline, giving sub-microsecond accuracy. `clock()` or `std::chrono` would measure host scheduling latency, not actual GPU execution time.

---

## Results (RTX 3050 Ti Laptop, 4096×4096)

| Kernel | Bandwidth | vs naive |
|--------|-----------|----------|
| `copy` (ceiling) | 179 GB/s | — |
| `transposeNaive` | 74 GB/s | 1× |
| `transposeCoalesced` | 178 GB/s | 2.4× |
| `transposeNoBankConflicts` | 178 GB/s | 2.4× |

The naive kernel wastes ~60% of bandwidth on non-coalesced writes. Both shared-memory kernels reach near-ceiling bandwidth. On this GPU (Ampere) bank conflicts are largely hidden, but the padding remains best practice and matters more on older architectures.

