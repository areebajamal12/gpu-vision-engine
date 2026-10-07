#!/usr/bin/env python3
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)

env = report["environment"]
workload = report["workload"]
print(f"\nGPU: {env['gpu_model']}  ")
if workload.get("operation") == "bilinear_resize":
    description = (f"bilinear resize {workload['input_width']}×{workload['input_height']} → "
                   f"{workload['output_width']}×{workload['output_height']}")
elif "kernel_size" in workload:
    description = f"Gaussian kernel {workload['kernel_size']}, sigma {workload['sigma']}"
elif workload.get("operation") == "lidar_voxelization":
    description = (f"real nuScenes sweep, {workload['input_points']} points, "
                   f"{workload['occupied_voxels']} occupied voxels")
else:
    description = workload["operation"]
if "width" in workload:
    description = f"{workload['width']}×{workload['height']}, {description}"
print(f"Workload: {description}\n")
is_voxelization = workload.get("operation") == "lidar_voxelization"
throughput_label = "Mpoints/s" if is_voxelization else "MP/s"
correctness_label = "Exact CPU match" if is_voxelization else "Max abs error"
print(f"| Implementation | Scope | Mean (ms) | p95 (ms) | {throughput_label} | "
      f"Mean speedup | p95 speedup | {correctness_label} |")
print("|---|---|---:|---:|---:|---:|---:|---:|")
for row in report["results"]:
    mean_speedup = row.get("mean_speedup_vs_naive", row.get("mean_speedup_vs_cpu"))
    p95_speedup = row.get("p95_speedup_vs_naive", row.get("p95_speedup_vs_cpu"))
    mean_speedup_text = f"{mean_speedup:.2f}×" if mean_speedup is not None else "—"
    p95_speedup_text = f"{p95_speedup:.2f}×" if p95_speedup is not None else "—"
    throughput = row.get("throughput_megapixels_per_second",
                         row.get("throughput_million_points_per_second"))
    correctness = "yes" if is_voxelization else f"{row['max_abs_error']:.6g}"
    print(f"| {row['implementation']} | {row['timing_scope']} | {row['mean_ms']:.3f} | "
          f"{row['p95_ms']:.3f} | {throughput:.1f} | "
          f"{mean_speedup_text} | {p95_speedup_text} | "
          f"{correctness} |")
