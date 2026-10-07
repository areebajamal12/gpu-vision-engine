# gpu-vision-engine

A C++17/CUDA library for hand-written image and LiDAR processing kernels, with CPU and OpenCV reference implementations. Milestones 1 and 2 implement baseline and shared-memory-tiled separable Gaussian blur for a 1920×1080 float32 grayscale image.

## Measured results

Correctness passed for every implementation before timing. These measurements are from a Kaggle Tesla T4 with a 1920×1080 float32 image, 9×9 kernel, sigma 2.0, 10 warm-ups, and 100 measured iterations. The saved evidence is [`benchmark-results/gaussian-blur-m2-t4.json`](benchmark-results/gaussian-blur-m2-t4.json).

| Implementation | Scope | Mean (ms) | p95 (ms) | Throughput (MP/s) | Mean speedup | p95 speedup |
|---|---|---:|---:|---:|---:|---:|
| CPU separable | Operation | 49.364 | 50.999 | 42.0 | — | — |
| OpenCV reference | Operation | 4.020 | 5.893 | 515.8 | — | — |
| CUDA naive | Kernel only | 0.364 | 0.427 | 5699.3 | 1.00× | 1.00× |
| CUDA naive | End to end | 4.402 | 4.711 | 471.0 | 1.00× | 1.00× |
| CUDA tiled | Kernel only | 0.280 | 0.312 | 7418.6 | 1.30× | 1.37× |
| CUDA tiled | End to end | 4.387 | 4.777 | 472.6 | 1.00× | 0.99× |

Shared-memory tiling reduced measured mean kernel-only latency by 23.2% (1.30× speedup). End-to-end mean latency was effectively unchanged because allocation and transfers dominate this baseline API path.

## Milestone 1 architecture

- `src/cpu`: C++17 horizontal and vertical 1D convolution passes.
- `src/cuda`: a baseline CUDA implementation with one thread per output pixel and separate horizontal/vertical kernels.
- `src/reference`: OpenCV `GaussianBlur` using the same border policy.
- `tests`: GoogleTest correctness and validation tests.
- `benchmarks`: correctness-gated timing and structured JSON reporting.

Inputs use `CV_32FC1`. All implementations use replicate borders. The CPU result must have maximum absolute error at most `1e-4` against OpenCV; CUDA must be at most `1e-3`. The benchmark exits without publishing results if either applicable gate fails.

### Shared-memory tiling

The Milestone 1 CUDA baseline reads every convolution sample directly from global memory. Neighboring output pixels therefore fetch many of the same input pixels repeatedly. The Milestone 2 kernels cooperatively load a block-sized tile plus its border halo into shared memory. Threads then reuse that on-chip tile for convolution. Loads across each warp remain contiguous where practical, while the separate horizontal and vertical passes preserve the original algorithm, workload, parameters, and border behavior. This design is expected to reduce redundant global-memory traffic; no performance improvement is claimed until a correctness-gated result file demonstrates it.

## Build and test

Requirements: CMake 3.24+, a C++17 compiler, and OpenCV development files. GoogleTest is fetched by CMake. CUDA is enabled automatically when `nvcc` is available; disable detection with `-DGVE_ENABLE_CUDA=OFF`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Benchmark protocol

The harness uses 10 warm-up iterations and 100 measured iterations. CPU operations use `std::chrono::steady_clock`. CUDA kernel-only measurements use CUDA events; end-to-end measurements use a host steady clock around allocation, host-to-device copies, kernel execution, synchronization, and device-to-host copies. JSON includes mean, p95, throughput, correctness error/tolerance, workload parameters, compiler, CPU/GPU identity, and CUDA runtime/driver versions.

```bash
./build/gaussian_blur_benchmark benchmark-results/gaussian-blur-m2-local.json
```

Generated JSON is ignored by default so unreviewed local results are not accidentally published. Force-add only verified evidence with `git add -f benchmark-results/<file>.json`.

## Reproduce on Kaggle (Tesla T4)

Create a Kaggle notebook with a T4 accelerator and Internet enabled, then run:

```bash
!bash <(curl -fsSL https://raw.githubusercontent.com/areebajamal12/gpu-vision-engine/main/kaggle/run_t4_benchmark.sh)
```

Download `/kaggle/working/gaussian-blur-m2-t4.json` after the run. The script verifies the GPU, builds CUDA, runs all tests, runs the correctness-gated benchmark, saves JSON, and prints a Markdown table. The JSON includes mean and p95 latency, throughput, and mean and p95 speedups relative to the matching naive CUDA timing scope.

## Roadmap

Future milestones may add additional image kernels and LiDAR operations. No performance claim is made for unmeasured work.

## Milestone 3: Sobel edge detection

Milestone 3 computes 3×3 Sobel horizontal and vertical gradients and combines them into a float32 gradient magnitude image. It includes C++17 CPU, OpenCV reference, naive CUDA, and shared-memory-tiled CUDA implementations. All use replicate borders on the same 1920×1080 workload.

The naive CUDA kernel reads each pixel's 3×3 neighborhood from global memory. The tiled kernel has a 32×8 thread block cooperatively load a 34×10 tile, including a one-pixel halo, into shared memory. Neighboring threads then reuse those values for both gradient directions, reducing repeated global reads while keeping contiguous loads where practical.

All implementations passed their correctness gates before timing: CPU output was within `1e-4` and each CUDA output was within `1e-3` maximum absolute error of OpenCV. The evidence is saved in [`benchmark-results/sobel-m3-t4.json`](benchmark-results/sobel-m3-t4.json).

| Implementation | Scope | Mean (ms) | p95 (ms) | Throughput (MP/s) | Mean speedup | p95 speedup |
|---|---|---:|---:|---:|---:|---:|
| CPU Sobel | Operation | 17.924 | 18.606 | 115.7 | — | — |
| OpenCV reference | Operation | 3.931 | 4.115 | 527.5 | — | — |
| CUDA naive | Kernel only | 0.114 | 0.138 | 18182.2 | 1.00× | 1.00× |
| CUDA naive | End to end | 3.824 | 3.971 | 542.3 | 1.00× | 1.00× |
| CUDA tiled | Kernel only | 0.116 | 0.131 | 17907.1 | 0.98× | 1.05× |
| CUDA tiled | End to end | 3.825 | 4.057 | 542.1 | 1.00× | 0.98× |

For this 3×3 stencil, shared-memory tiling did not improve mean latency on the T4: tiled kernel-only mean was about 1.5% slower, while its p95 was about 5.2% lower. End-to-end latency was effectively unchanged and dominated by allocation and transfers.

Reproduce the complete T4 workflow with:

```bash
!bash <(curl -fsSL https://raw.githubusercontent.com/areebajamal12/gpu-vision-engine/main/kaggle/run_t4_sobel_benchmark.sh)
```

The result is written to `/kaggle/working/sobel-m3-t4.json`.

## Milestone 4: bilinear image resize

Milestone 4 resizes a 1920×1080 float32 grayscale image to 1280×720 using bilinear interpolation. It includes C++17 CPU, OpenCV reference, naive CUDA, and texture-backed CUDA implementations.

The naive CUDA kernel calculates four source samples and interpolation weights for each output pixel using ordinary global-memory reads. The optimized path uses a CUDA texture object, which provides a cache designed for spatial image access and performs bilinear sampling in hardware. Texture interpolation has limited fractional precision, so its documented OpenCV maximum-absolute-error tolerance is `5e-3`; CPU and naive CUDA use `1e-4`.

Resize results are not published yet. The benchmark refuses to save timing results unless every applicable correctness gate passes. Run the complete T4 workflow with:

```bash
!bash <(curl -fsSL https://raw.githubusercontent.com/areebajamal12/gpu-vision-engine/main/kaggle/run_t4_resize_benchmark.sh)
```

The result is written to `/kaggle/working/resize-m4-t4.json`. No resize performance claim will be added until that measured evidence is committed.
