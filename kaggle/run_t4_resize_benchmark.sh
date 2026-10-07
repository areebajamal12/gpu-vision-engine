#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_URL="${REPOSITORY_URL:-https://github.com/areebajamal12/gpu-vision-engine.git}"
WORK_DIR="${WORK_DIR:-/kaggle/working/gpu-vision-engine-resize}"
RESULT_PATH="${RESULT_PATH:-/kaggle/working/resize-m4-t4.json}"

echo "## GPU environment"
nvidia-smi
nvcc --version

apt-get update -qq
apt-get install -y -qq libopencv-dev

git clone --depth 1 "$REPOSITORY_URL" "$WORK_DIR"
cmake -S "$WORK_DIR" -B "$WORK_DIR/build" -DCMAKE_BUILD_TYPE=Release -DGVE_ENABLE_CUDA=ON
cmake --build "$WORK_DIR/build" --parallel
ctest --test-dir "$WORK_DIR/build" --output-on-failure
"$WORK_DIR/build/resize_benchmark" "$RESULT_PATH"
python3 "$WORK_DIR/kaggle/results_table.py" "$RESULT_PATH"

echo
echo "Saved measured evidence to: $RESULT_PATH"
