# gpu-vision-engine

A C++17/CUDA library for hand-written image and LiDAR processing kernels, with CPU and OpenCV reference implementations. The implemented milestones cover Gaussian blur, Sobel edge detection, and bilinear image resize.

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

All implementations passed their correctness gates before timing. The measured evidence is saved in [`benchmark-results/resize-m4-t4.json`](benchmark-results/resize-m4-t4.json).

| Implementation | Scope | Mean (ms) | p95 (ms) | Throughput (MP/s) | Mean speedup | p95 speedup |
|---|---|---:|---:|---:|---:|---:|
| CPU bilinear | Operation | 8.817 | 9.173 | 104.5 | — | — |
| OpenCV reference | Operation | 0.746 | 0.874 | 1235.8 | — | — |
| CUDA naive | Kernel only | 0.054 | 0.058 | 16995.2 | 1.00× | 1.00× |
| CUDA naive | End to end | 2.800 | 2.900 | 329.1 | 1.00× | 1.00× |
| CUDA texture | Kernel only | 0.055 | 0.057 | 16730.4 | 0.98× | 1.02× |
| CUDA texture | End to end | 2.780 | 2.859 | 331.5 | 1.01× | 1.01× |

On this T4 workload, the texture path did not improve mean kernel-only latency: it was about 1.6% slower than the naive kernel. End-to-end mean was about 0.7% lower, while transfers and allocation dominated both paths.

Run the complete T4 workflow with:

```bash
!bash <(curl -fsSL https://raw.githubusercontent.com/areebajamal12/gpu-vision-engine/main/kaggle/run_t4_resize_benchmark.sh)
```

The result is written to `/kaggle/working/resize-m4-t4.json`.

## Milestone 5: nuScenes LiDAR voxelization

Milestone 5 consumes a real nuScenes LiDAR `.pcd.bin` sweep. Each five-float record contains x, y, z, intensity, and ring index; voxel statistics use x, y, z, and intensity. The benchmark does not generate or silently substitute synthetic data.

The grid covers x and y in `[-50, 50)` metres and z in `[-5, 3)` metres, with `0.25 × 0.25 × 0.20` metre voxels. Non-finite points and points outside the half-open bounds are rejected. Occupied voxels are sorted by their linear z-y-x index and contain a point count plus fixed-point sums of x, y, z, and intensity. Integer aggregation makes the result independent of reduction order.

The CPU path maps points, stable-sorts them by voxel index, and reduces adjacent records. CUDA follows the same race-free design: a mapping kernel produces keys and fixed-point features, then stable key sorting and segmented reduction produce deterministic sparse voxel statistics. The benchmark requires an exact CUDA-to-CPU match before it writes timing evidence.

All implementations passed their correctness gates before timing, including an exact CUDA-to-CPU voxel output match. The measured evidence is saved in [`benchmark-results/voxelization-m5-t4.json`](benchmark-results/voxelization-m5-t4.json). The real sweep contained 34,720 input points: 30,575 were inside the grid, 4,145 were rejected, and 8,057 voxels were occupied.

| Implementation | Scope | Mean (ms) | p95 (ms) | Throughput (Mpoints/s) | Mean speedup vs CPU | p95 speedup vs CPU |
|---|---|---:|---:|---:|---:|---:|
| CPU sort/reduce | Operation | 2.106 | 2.304 | 16.5 | — | — |
| CUDA sort/reduce | Kernel only | 0.396 | 0.414 | 87.7 | 5.32× | 5.57× |
| CUDA sort/reduce | End to end | 1.048 | 1.140 | 33.1 | 2.01× | 2.02× |

On this Tesla T4 run, CUDA reduced mean voxelization latency by 5.32× for GPU work alone and 2.01× including allocation, transfers, and result download. To reproduce the one-shot workflow, attach a licensed nuScenes dataset to a Kaggle notebook with a Tesla T4 and Internet enabled, then run:

```bash
!bash <(curl -fsSL https://raw.githubusercontent.com/areebajamal12/gpu-vision-engine/main/kaggle/run_t4_voxelization_benchmark.sh)
```

The script finds the first `samples/LIDAR_TOP/*.pcd.bin` sweep under `/kaggle/input`, runs all tests, performs 10 warm-ups and 100 measured iterations, and writes `/kaggle/working/voxelization-m5-t4.json`. The result records the sweep filename, SHA-256, input/valid/rejected point counts, occupied voxel count, grid definition, environment, timing methodology, mean, p95, throughput, and speedup.
