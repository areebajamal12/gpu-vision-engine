# gpu-vision-engine

A C++17/CUDA library of hand-written image and LiDAR kernels, built around correctness-gated CPU/OpenCV references and reproducible Tesla T4 benchmarks.

The project covers separable Gaussian blur, Sobel edge detection, bilinear resize, and deterministic voxelization of real nuScenes LiDAR sweeps. CUDA is enabled automatically when `nvcc` is available; the same tree builds CPU-only on macOS.

## Performance highlights

Every number below comes from a committed structured JSON result. Each benchmark completed 10 warm-up runs and 100 measured iterations, and timing was accepted only after its correctness gate passed.

| Workload | Best measured result | What it demonstrates | Evidence |
|---|---|---|---|
| Gaussian blur, 1920×1080 float32 | Tiled CUDA: **0.280 ms mean kernel latency, 1.30× faster** than naive CUDA | Shared-memory halo reuse reduced redundant global reads | [JSON](benchmark-results/gaussian-blur-m2-t4.json) |
| Real nuScenes LiDAR sweep, 34,720 points | CUDA: **0.396 ms kernel, 5.32× CPU speedup**; **1.048 ms end to end, 2.01× speedup** | Deterministic sparse GPU sort/reduce retained a system-level gain | [JSON](benchmark-results/voxelization-m5-t4.json) |
| Sobel, 1920×1080 float32 | Tiled kernel mean was **0.98×** the naive baseline | A small 3×3 stencil did not benefit from explicit shared-memory tiling on this T4 | [JSON](benchmark-results/sobel-m3-t4.json) |
| Bilinear resize, 1920×1080 → 1280×720 | Texture kernel mean was **0.98×** the naive baseline | Hardware texture sampling did not automatically improve this workload | [JSON](benchmark-results/resize-m4-t4.json) |

Measured on a Kaggle Tesla T4. Speedups compare matching timing scopes except LiDAR, where the CPU operation is compared with CUDA kernel-only and CUDA end-to-end measurements as labeled.

## Key engineering findings

- Optimization is workload-specific. Shared-memory tiling helped the 9×9 separable Gaussian blur, but not the 3×3 Sobel kernel; texture sampling did not improve resize kernel mean.
- Kernel speed is not application speed. Gaussian blur improved from 0.364 ms to 0.280 ms in the kernel, while end-to-end latency stayed near 4.4 ms because allocation and PCIe transfers dominated the baseline API.
- LiDAR voxelization preserved a system-level benefit. Its 5.32× kernel-only speedup remained 2.01× after allocation, transfers, sorting/reduction, and result download.
- Correctness is part of benchmarking. Image kernels are checked against OpenCV within documented tolerances. CUDA voxelization must exactly match deterministic CPU voxel keys, counts, and fixed-point feature sums.
- Reproducibility is recorded, not implied. Result JSON includes hardware, CUDA/compiler versions and flags, workload parameters, iteration counts, timing method, correctness data, and dataset identity where applicable.

## Architecture

```text
include/gve/       Public C++ API and shared data structures
src/cpu/           C++17 baselines and deterministic reference algorithms
src/cuda/          Naive and optimized hand-written CUDA implementations
src/reference/     OpenCV image references
benchmarks/        Correctness-gated timing and JSON reporting
tests/             GoogleTest unit, boundary, and CPU/CUDA agreement tests
viewer/            Optional OpenGL nuScenes point/voxel viewer
kaggle/            One-shot Tesla T4 reproduction workflows
benchmark-results/ Reviewed measured evidence committed explicitly
```

Image kernels operate on single-channel float32 data. Gaussian blur uses two 1D convolution passes; its optimized kernels stage tiles and halo pixels in shared memory. Sobel uses either direct 3×3 reads or a shared tile. Resize follows OpenCV half-pixel bilinear coordinates, with global-memory and texture-backed CUDA paths.

LiDAR input uses the native nuScenes five-float record (`x`, `y`, `z`, intensity, ring index). The grid spans x/y `[-50, 50)` metres and z `[-5, 3)` metres with `0.25 × 0.25 × 0.20` metre voxels. CPU and CUDA map valid points to linear voxel keys, stable-sort, and reduce adjacent keys. Aggregated coordinates are quantized to millimetres and intensity to `1e-4`, making integer sums deterministic and reduction-order independent.

## Measured evidence

### Gaussian blur

Workload: 1920×1080 float32, 9×9 kernel, sigma 2.0, replicate border. Evidence: [gaussian-blur-m2-t4.json](benchmark-results/gaussian-blur-m2-t4.json).

| Implementation | Scope | Mean (ms) | p95 (ms) | Throughput (MP/s) | Mean speedup |
|---|---|---:|---:|---:|---:|
| CPU separable | Operation | 49.364 | 50.999 | 42.0 | — |
| OpenCV reference | Operation | 4.020 | 5.893 | 515.8 | — |
| CUDA naive | Kernel only | 0.364 | 0.427 | 5699.3 | 1.00× |
| CUDA naive | End to end | 4.402 | 4.711 | 471.0 | 1.00× |
| CUDA tiled | Kernel only | 0.280 | 0.312 | 7418.6 | 1.30× |
| CUDA tiled | End to end | 4.387 | 4.777 | 472.6 | 1.00× |

### Real nuScenes LiDAR voxelization

The measured sweep contained 34,720 points; 30,575 were valid, 4,145 were rejected, and 8,057 voxels were occupied. CUDA exactly matched CPU output. Evidence: [voxelization-m5-t4.json](benchmark-results/voxelization-m5-t4.json).

| Implementation | Scope | Mean (ms) | p95 (ms) | Throughput (Mpoints/s) | Mean speedup vs CPU |
|---|---|---:|---:|---:|---:|
| CPU sort/reduce | Operation | 2.106 | 2.304 | 16.5 | — |
| CUDA sort/reduce | Kernel only | 0.396 | 0.414 | 87.7 | 5.32× |
| CUDA sort/reduce | End to end | 1.048 | 1.140 | 33.1 | 2.01× |

### Sobel edge detection

Workload: 1920×1080 float32, 3×3 Sobel magnitude, replicate border. Evidence: [sobel-m3-t4.json](benchmark-results/sobel-m3-t4.json).

| Implementation | Scope | Mean (ms) | p95 (ms) | Throughput (MP/s) | Mean speedup |
|---|---|---:|---:|---:|---:|
| CPU Sobel | Operation | 17.924 | 18.606 | 115.7 | — |
| OpenCV reference | Operation | 3.931 | 4.115 | 527.5 | — |
| CUDA naive | Kernel only | 0.114 | 0.138 | 18182.2 | 1.00× |
| CUDA naive | End to end | 3.824 | 3.971 | 542.3 | 1.00× |
| CUDA tiled | Kernel only | 0.116 | 0.131 | 17907.1 | 0.98× |
| CUDA tiled | End to end | 3.825 | 4.057 | 542.1 | 1.00× |

The tiled kernel was about 1.5% slower by mean, although its p95 was lower. The result is retained because unsuccessful optimization hypotheses are useful engineering evidence.

### Bilinear resize

Workload: 1920×1080 to 1280×720 float32. Evidence: [resize-m4-t4.json](benchmark-results/resize-m4-t4.json).

| Implementation | Scope | Mean (ms) | p95 (ms) | Throughput (MP/s) | Mean speedup |
|---|---|---:|---:|---:|---:|
| CPU bilinear | Operation | 8.817 | 9.173 | 104.5 | — |
| OpenCV reference | Operation | 0.746 | 0.874 | 1235.8 | — |
| CUDA naive | Kernel only | 0.054 | 0.058 | 16995.2 | 1.00× |
| CUDA naive | End to end | 2.800 | 2.900 | 329.1 | 1.00× |
| CUDA texture | Kernel only | 0.055 | 0.057 | 16730.4 | 0.98× |
| CUDA texture | End to end | 2.780 | 2.859 | 331.5 | 1.01× |

## Interactive nuScenes viewer

The optional OpenGL viewer makes the LiDAR pipeline tangible without adding a dependency to the library, tests, benchmarks, or CI. It loads a real nuScenes `.pcd.bin` sweep and overlays intensity-colored input points with occupancy-colored voxel centers. When CUDA is detected, those voxel statistics are produced by the same deterministic CUDA implementation used by the benchmark; CPU voxelization is the portable fallback.

```bash
cmake -S . -B build-viewer -DCMAKE_BUILD_TYPE=Release \
  -DGVE_BUILD_TESTS=OFF \
  -DGVE_BUILD_BENCHMARKS=OFF -DGVE_BUILD_VIEWER=ON
cmake --build build-viewer --parallel
./build-viewer/voxel_viewer /path/to/samples/LIDAR_TOP/<sweep>.pcd.bin
```

GLFW is fetched only when `GVE_BUILD_VIEWER=ON`; OpenGL must be available on the host. Drag with the left mouse button to orbit, scroll to zoom, press `1` for points, `2` for voxels, `3` for both, and `R` to reset. Press `S` to save a clean `voxel-viewer.ppm` frame for a README image or portfolio asset. On macOS without CUDA, add `-DGVE_ENABLE_CUDA=OFF` explicitly if needed.

The in-window overlay labels the active voxelization backend and displays the verified Tesla T4 CPU, CUDA kernel-only, and end-to-end measurements from [`voxelization-m5-t4.json`](benchmark-results/voxelization-m5-t4.json). These are evidence annotations—not measurements performed by the viewer. The viewer remains excluded from headless CI and adds no visualization benchmark claims.

## Build and test

Requirements: CMake 3.24+, a C++17 compiler, and OpenCV development files. GoogleTest is fetched by CMake. CUDA is detected automatically; explicitly disable it with `-DGVE_ENABLE_CUDA=OFF`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The default build does not fetch or compile the viewer. macOS therefore remains a CPU/OpenCV build, while a CUDA environment adds the GPU library and GPU correctness tests.

## Benchmark methodology

CPU operations use `std::chrono::steady_clock`. CUDA kernel-only measurements use CUDA events around GPU work. End-to-end measurements use a host steady clock around allocation, host-to-device copies, kernel work and synchronization, and device-to-host copies. Mean and p95 are calculated over 100 measured iterations after 10 warm-ups.

Kernel-only timing answers whether the GPU algorithm itself improved. End-to-end timing answers whether that improvement survives data movement and setup costs. Both are reported because a fast isolated kernel can still be a poor system-level tradeoff.

Generated result files are ignored by default. Only reviewed evidence should be committed with `git add -f benchmark-results/<file>.json`.

## Reproduction on Kaggle T4

Create a Kaggle notebook with a T4 accelerator and Internet enabled, then run the workflow for the desired workload:

```bash
# Gaussian blur
!bash <(curl -fsSL https://raw.githubusercontent.com/areebajamal12/gpu-vision-engine/main/kaggle/run_t4_benchmark.sh)

# Sobel
!bash <(curl -fsSL https://raw.githubusercontent.com/areebajamal12/gpu-vision-engine/main/kaggle/run_t4_sobel_benchmark.sh)

# Resize
!bash <(curl -fsSL https://raw.githubusercontent.com/areebajamal12/gpu-vision-engine/main/kaggle/run_t4_resize_benchmark.sh)

# LiDAR voxelization; attach a licensed nuScenes dataset first
!bash <(curl -fsSL https://raw.githubusercontent.com/areebajamal12/gpu-vision-engine/main/kaggle/run_t4_voxelization_benchmark.sh)
```

Each script verifies the GPU, configures a CUDA release build, runs the complete test suite, executes the correctness-gated benchmark, saves JSON under `/kaggle/working`, and prints a Markdown results table.
