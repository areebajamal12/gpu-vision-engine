# gpu-vision-engine

A C++17/CUDA library for hand-written image and LiDAR processing kernels, with CPU and OpenCV reference implementations. Milestone 1 implements separable Gaussian blur for a 1920×1080 float32 grayscale image.

## Measured results

No benchmark results are published yet. Results will be added only after the correctness gate passes on the measured system, with the generated JSON committed as evidence. The Kaggle workflow below is prepared for a Tesla T4 run.

## Milestone 1 architecture

- `src/cpu`: C++17 horizontal and vertical 1D convolution passes.
- `src/cuda`: a baseline CUDA implementation with one thread per output pixel and separate horizontal/vertical kernels.
- `src/reference`: OpenCV `GaussianBlur` using the same border policy.
- `tests`: GoogleTest correctness and validation tests.
- `benchmarks`: correctness-gated timing and structured JSON reporting.

Inputs use `CV_32FC1`. All implementations use replicate borders. The CPU result must have maximum absolute error at most `1e-4` against OpenCV; CUDA must be at most `1e-3`. The benchmark exits without publishing results if either applicable gate fails.

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
./build/gaussian_blur_benchmark benchmark-results/gaussian-blur-local.json
```

Generated JSON is ignored by default so unreviewed local results are not accidentally published. Force-add only verified evidence with `git add -f benchmark-results/<file>.json`.

## Reproduce on Kaggle (Tesla T4)

Create a Kaggle notebook with a T4 accelerator and Internet enabled, then run:

```bash
!bash <(curl -fsSL https://raw.githubusercontent.com/areebajamal12/gpu-vision-engine/main/kaggle/run_t4_benchmark.sh)
```

Download `/kaggle/working/gaussian-blur-t4.json` after the run. The script verifies the GPU, builds CUDA, runs all tests, runs the correctness-gated benchmark, saves JSON, and prints a Markdown table.

## Roadmap

Future milestones may add optimized memory access, shared-memory tiling, additional image kernels, and LiDAR operations. No performance claim is made for unmeasured work.
