#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_URL="${REPOSITORY_URL:-https://github.com/areebajamal12/gpu-vision-engine.git}"
WORK_DIR="${WORK_DIR:-/kaggle/working/gpu-vision-engine-voxelization}"
RESULT_PATH="${RESULT_PATH:-/kaggle/working/voxelization-m5-t4.json}"

echo "## GPU environment"
nvidia-smi
nvcc --version

if [[ -z "${SWEEP_PATH:-}" ]]; then
  SWEEP_PATH="$(find /kaggle/input -type f -path '*/samples/LIDAR_TOP/*.pcd.bin' -print -quit)"
fi
if [[ -z "${SWEEP_PATH:-}" || ! -f "$SWEEP_PATH" ]]; then
  echo "No real nuScenes sweep found." >&2
  echo "Attach a nuScenes dataset containing samples/LIDAR_TOP/*.pcd.bin, or set SWEEP_PATH." >&2
  exit 2
fi
if [[ "$SWEEP_PATH" != *.pcd.bin ]]; then
  echo "SWEEP_PATH must identify a nuScenes .pcd.bin LiDAR sweep." >&2
  exit 2
fi

echo "Using real nuScenes sweep: $SWEEP_PATH"
export NUSCENES_SWEEP_SHA256
NUSCENES_SWEEP_SHA256="$(sha256sum "$SWEEP_PATH" | cut -d' ' -f1)"

apt-get update -qq
apt-get install -y -qq libopencv-dev
git clone --depth 1 "$REPOSITORY_URL" "$WORK_DIR"
cmake -S "$WORK_DIR" -B "$WORK_DIR/build" -DCMAKE_BUILD_TYPE=Release -DGVE_ENABLE_CUDA=ON
cmake --build "$WORK_DIR/build" --parallel
ctest --test-dir "$WORK_DIR/build" --output-on-failure
"$WORK_DIR/build/voxelization_benchmark" "$SWEEP_PATH" "$RESULT_PATH"
python3 "$WORK_DIR/kaggle/results_table.py" "$RESULT_PATH"

echo
echo "Saved measured evidence to: $RESULT_PATH"
