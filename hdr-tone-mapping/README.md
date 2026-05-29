# HDR Tone Mapping — CUDA Assignment

Extension of the `cuda-webcam-filter` template adding real-time HDR tone mapping as a new filter type. Three algorithms are implemented — Global Reinhard, ACES Filmic, and Local Reinhard — each with a GPU CUDA kernel and a CPU reference, displayed side-by-side with per-frame timing.

Benchmarks run on an **NVIDIA GeForce RTX 3050 Ti Laptop GPU** (CUDA 13.2, Nsight Systems 2025.6.3).

## Build & Run

Run all commands inside the **x64 Native Tools Command Prompt for VS 2022** so `nvcc` and `cmake` are on `PATH`.

### 1. Configure

```cmd
cd assigments\hdr-tone-mapping\cuda-webcam-filter
mkdir build
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="C:\opencv\opencv\build\x64\vc16\lib" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

> **Notes:**
> - Use `"Ninja"` generator — VS 2026 does not ship the CUDA toolset integration required by the Visual Studio generators.
> - Point `OpenCV_DIR` to the `x64\vc16\lib` subfolder, not the top-level build dir, to bypass the version compatibility check.
> - `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` silences the CMake 4.x deprecation error from the bundled `plog` dependency.

### 2. Build

```cmd
cmake --build .
```

### 3. Add OpenCV DLLs to PATH (required at runtime)

```cmd
set PATH=C:\opencv\opencv\build\x64\vc16\bin;%PATH%
```

### 4. Run

```cmd
# Global Reinhard — webcam
cuda-webcam-filter.exe -f hdr --hdr-algo reinhard --exposure 2.5 --preview

# ACES Filmic — synthetic gradient
cuda-webcam-filter.exe -f hdr --hdr-algo aces -i synthetic -s gradient --exposure 3.0 --preview

# Local Reinhard (shared memory) — synthetic checkerboard
cuda-webcam-filter.exe -f hdr --hdr-algo local -i synthetic -s checkerboard --exposure 2.5 --gamma 2.2 --saturation 1.2 --preview

# Standard convolution filter (unchanged)
cuda-webcam-filter.exe -f blur -i synthetic -s gradient --preview
cuda-webcam-filter.exe -f edge -i synthetic -s checkerboard --preview
```

Press **ESC** to exit.

---

## System Specs

**Host**

| Property   | Value                                        |
| ---------- | -------------------------------------------- |
| OS         | Microsoft Windows 11 Home (Build 10.0.26200) |
| CPU        | 11th Gen Intel Core i5-11300H @ 3.10 GHz     |
| System RAM | 32 GB                                        |

**GPU**

| Property                     | Value                                 |
| ---------------------------- | ------------------------------------- |
| GPU                          | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| Architecture                 | Ampere (Compute Capability 8.6)       |
| Streaming Multiprocessors    | 20 SMs                                |
| VRAM                         | 4096 MiB (4.0 GB)                     |
| Max SM Clock                 | 2100 MHz                              |
| Max Memory Clock             | 6001 MHz                              |
| Memory Bus Width             | 128-bit                               |
| Theoretical Memory Bandwidth | ~192 GB/s                             |
| PCIe Interface               | Gen 3 × 16                            |
| NVIDIA Driver                | 595.71                                |
| CUDA Version                 | 13.2                                  |
| Nsight Systems               | 2025.6.3                              |

---

## What is HDR Tone Mapping?

Real-world scenes have a dynamic range of up to 100,000:1 — direct sunlight vs deep shadow. A standard 8-bit display can only represent a ratio of 255:1. **Tone mapping** compresses the scene's luminance range into the displayable range while preserving local contrast and perceived colour.

The key quantity is **luminance** computed with Rec. 709 coefficients:

```
L = 0.2126·R + 0.7152·G + 0.0722·B
```

Each algorithm maps scene luminance `L` (after an exposure pre-scale) to a display value `L_out ∈ [0, 1]`, then the scale factor `L_out / L` is applied to all channels to preserve hue.

---

## Command-Line Parameters

| Flag            | Default    | Description                                  |
| --------------- | ---------- | -------------------------------------------- |
| `-f hdr`        | —          | Selects the HDR tone mapping filter          |
| `--hdr-algo`    | `reinhard` | Algorithm: `reinhard`, `aces`, `local`       |
| `--exposure`    | `2.5`      | Linear pre-scale applied before tone mapping |
| `--gamma`       | `2.2`      | Display gamma correction (applied last)      |
| `--saturation`  | `1.2`      | Colour saturation boost (1.0 = neutral)      |
| `--white-point` | `4.0`      | Reinhard white point `L_white`               |

---

## Implementation Breakdown

### 1. Filter Framework Extension

`FilterType::HDR_TONEMAPPING` was added to the existing enum in `filter_utils.h` alongside `BLUR`, `SHARPEN`, `EDGE_DETECTION`, `EMBOSS`, and `IDENTITY`. The string `"hdr"` maps to it via `stringToFilterType()`. Since HDR is not a convolution filter, `createFilterKernel()` returns an identity kernel for this type; `main.cpp` detects `isHDR` and routes to the dedicated HDR path instead.

---

### 2. Luminance and Colour Helpers — `hdr_bgr2lum`, `hdr_satGamma`

```cuda
__host__ __device__ float hdr_bgr2lum(float b, float g, float r) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;  // Rec. 709, BGR channel order
}

__host__ __device__ void hdr_satGamma(float &c0, float &c1, float &c2,
                                       float sat, float invGamma) {
    float lum = hdr_bgr2lum(c0, c1, c2);
    c0 = lum + sat * (c0 - lum);   // saturation blend toward grey
    c1 = lum + sat * (c1 - lum);
    c2 = lum + sat * (c2 - lum);
    c0 = powf(c0 < 0 ? 0 : c0, invGamma);  // gamma correction
    ...
}
```

OpenCV frames are in **BGR** order, so the luminance coefficients are swapped relative to RGB (B uses 0.0722, R uses 0.2126). Both helpers are `__host__ __device__` so the same code is shared between GPU kernels and the CPU reference — no separate implementations.

---

### 3. Algorithm 0 — Global Reinhard (`k_hdr_reinhardGlobal`)

Extended Reinhard operator (Reinhard et al. 2002):

```
L_out = L · (1 + L / L_white²) / (1 + L)
scale = L_out / L
R_out = R · scale,  G_out = G · scale,  B_out = B · scale
```

The white-point term `L / L_white²` gradually compresses highlights above `L_white` to zero, preventing blown-out clipping while keeping mid-tones almost linear.

```cuda
__global__ void k_hdr_reinhardGlobal(const uchar *in, uchar *out,
                                      int W, int H,
                                      float exposure, float white2,
                                      float sat, float invGamma) {
    int i = (y * W + x) * 3;
    float c0 = in[i+0] / 255.f * exposure;   // expose + normalise
    float L   = hdr_bgr2lum(c0, c1, c2);
    float L_out = L * (1.f + L / white2) / (1.f + L);
    float scale = (L > 1e-6f) ? L_out / L : 0.f;
    c0 *= scale; c1 *= scale; c2 *= scale;
    hdr_satGamma(c0, c1, c2, sat, invGamma);
    out[i+0] = hdr_toUchar(c0);
    ...
}
```

One thread per pixel; fully fused (no intermediate buffers). Memory access is coalesced — consecutive threads process consecutive pixels in the same row.

---

### 4. Algorithm 1 — ACES Filmic (`k_hdr_aces`)

ACES RRT+ODT approximation (Narkowicz 2015). Applies a rational S-curve per channel:

```
f(x) = (x · (2.51x + 0.03)) / (x · (2.43x + 0.59) + 0.14)
```

The S-curve naturally lifts shadows, compresses highlights, and keeps mid-tones near linear — matching the look of cinema film. Applied per-channel rather than to luminance only, which shifts hue slightly in saturated highlights (intentional filmic look).

```cuda
__host__ __device__ float hdr_acesFilmic(float x) {
    return clamp01((x * (2.51f*x + 0.03f)) / (x * (2.43f*x + 0.59f) + 0.14f));
}

__global__ void k_hdr_aces(...) {
    float c0 = hdr_acesFilmic(in[i+0] / 255.f * exposure);
    float c1 = hdr_acesFilmic(in[i+1] / 255.f * exposure);
    float c2 = hdr_acesFilmic(in[i+2] / 255.f * exposure);
    hdr_satGamma(c0, c1, c2, sat, invGamma);
    ...
}
```

---

### 5. Algorithm 2 — Local Reinhard with Shared Memory (`k_hdr_localReinhard`)

Local tone mapping adapts to the luminance of each pixel's neighbourhood rather than the global average. This preserves local contrast in high-frequency regions (edges, textures) while still compressing the global range.

**Two-pass pipeline:**

1. `k_hdr_toFloat` — converts `uint8 + exposure` to a `float` buffer in one kernel.
2. `k_hdr_localReinhard` — loads a halo tile into shared memory, averages a 9×9 window, applies `scale = 1 / (1 + L_local_avg)`.

**Shared memory tiling:**

```cuda
#define HDR_TILE_W  16
#define HDR_TILE_H  16
#define HDR_HALO     4          // 9×9 neighbourhood
#define HDR_SH_W    (16 + 2*4)  // 24
#define HDR_SH_H    (16 + 2*4)  // 24

__shared__ float shLum[HDR_SH_H][HDR_SH_W];  // 24×24×4 = 2304 bytes/block

// Cooperative halo load: all threads fill the 24×24 tile
for (int dy = ty; dy < HDR_SH_H; dy += HDR_TILE_H)
    for (int dx = tx; dx < HDR_SH_W; dx += HDR_TILE_W) {
        int gx = clamp(bx + dx - HDR_HALO, 0, W-1);
        shLum[dy][dx] = hdr_bgr2lum(in[gx, gy]);
    }
__syncthreads();

// 9×9 sum entirely from shared memory — no global reads in inner loop
float sumL = 0;
for (int dy = 0; dy < 9; dy++)
    for (int dx = 0; dx < 9; dx++)
        sumL += shLum[ty + dy][tx + dx];

float scale = 1.f / (1.f + sumL / 81.f);
```

Without shared memory each pixel would read 81 luminance values from global memory; with tiling each value is loaded once and reused by up to 81 threads — reducing global traffic by ~81× in the inner loop.

---

### 6. GPU Wrapper — `applyHDRFilterGPU`

```cuda
void applyHDRFilterGPU(const cv::Mat &input, cv::Mat &output,
                        const HdrOptions &opts) {
    cudaMalloc(&d_in, imgBytes);
    cudaMemcpy(d_in, input.data, imgBytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(divUp(W, 16), divUp(H, 16));

    if      (opts.algorithm == "aces")  k_hdr_aces<<<grid, block>>>(...);
    else if (opts.algorithm == "local") { k_hdr_toFloat<<<...>>>; k_hdr_localReinhard<<<...>>>; }
    else                                k_hdr_reinhardGlobal<<<grid, block>>>(...);

    cudaGetLastError();
    cudaDeviceSynchronize();
    cudaMemcpy(output.data, d_out, imgBytes, cudaMemcpyDeviceToHost);
    ...
}
```

Follows the same pattern as the existing `applyFilterGPU` so it integrates cleanly into the per-frame loop in `main.cpp`.

---

### 7. CPU Reference — `applyHDRFilterCPU`

Implements all three algorithms with the same `__host__ __device__` helpers as the GPU path. This ensures CPU and GPU apply identical maths — any pixel difference larger than ±1 (float rounding) indicates a kernel bug.

---

### 8. Error Checking — `CHECK_CUDA_ERROR`

```cuda
#define CHECK_CUDA_ERROR(call) {                                   \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        PLOG_ERROR << "CUDA error in " #call ": "                  \
                   << cudaGetErrorString(err);                     \
        return;                                                    \
    }                                                              \
}
```

Every CUDA API call is wrapped. Without this, a failed `cudaMalloc` or a kernel launched with the wrong architecture would silently produce a zero-filled output frame.

---

## Results (RTX 3050 Ti Laptop, 1280×720)

| Algorithm                        | CPU (ms) | GPU (ms) | Speedup | Max pixel diff |
| -------------------------------- | -------- | -------- | ------- | -------------- |
| Global Reinhard                  | 48.2     | 0.144    | 333×    | 0              |
| ACES Filmic                      | 45.2     | 0.133    | 340×    | 0              |
| Local Reinhard (shared mem, 9×9) | 126.4    | 0.404    | 313×    | 1              |

- **Global / ACES** — zero pixel difference; GPU and CPU produce bit-identical output within `unsigned char` rounding.
- **Local Reinhard** — max diff of 1 LSB due to different float accumulation order in the 81-element sum; not a bug.
- **Local Reinhard CPU** is ~2.8× slower than the global algorithms because of the 81 extra memory reads per pixel with no SIMD.
- **Local Reinhard GPU** is ~3× slower than the global kernels due to the two-pass pipeline (extra `float` buffer write + read), but still well under 1 ms per frame.

All three algorithms sustain real-time frame rates (>30 FPS on HD input) on the GPU.
