#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_URL="${REPOSITORY_URL:-https://github.com/areebajamal12/gpu-vision-engine.git}"
RESULT_PATH="${RESULT_PATH:-/kaggle/working/gaussian-blur-m2-t4.json}"

echo "## GPU environment"
nvidia-smi
nvcc --version

apt-get update -qq
apt-get install -y -qq libopencv-dev

git clone --depth 1 "$REPOSITORY_URL" /kaggle/working/gpu-vision-engine
cmake -S /kaggle/working/gpu-vision-engine -B /kaggle/working/gpu-vision-engine/build \
  -DCMAKE_BUILD_TYPE=Release -DGVE_ENABLE_CUDA=ON
cmake --build /kaggle/working/gpu-vision-engine/build --parallel
ctest --test-dir /kaggle/working/gpu-vision-engine/build --output-on-failure
/kaggle/working/gpu-vision-engine/build/gaussian_blur_benchmark "$RESULT_PATH"

python3 /kaggle/working/gpu-vision-engine/kaggle/results_table.py "$RESULT_PATH"
echo
echo "Saved measured evidence to: $RESULT_PATH"
