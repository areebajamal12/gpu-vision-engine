#!/usr/bin/env python3
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)

env = report["environment"]
workload = report["workload"]
print(f"\nGPU: {env['gpu_model']}  ")
print(f"Workload: {workload['width']}×{workload['height']}, kernel {workload['kernel_size']}, sigma {workload['sigma']}\n")
print("| Implementation | Scope | Mean (ms) | p95 (ms) | MP/s | Mean speedup | p95 speedup | Max abs error |")
print("|---|---|---:|---:|---:|---:|---:|---:|")
for row in report["results"]:
    mean_speedup = row.get("mean_speedup_vs_naive")
    p95_speedup = row.get("p95_speedup_vs_naive")
    mean_speedup_text = f"{mean_speedup:.2f}×" if mean_speedup is not None else "—"
    p95_speedup_text = f"{p95_speedup:.2f}×" if p95_speedup is not None else "—"
    print(f"| {row['implementation']} | {row['timing_scope']} | {row['mean_ms']:.3f} | "
          f"{row['p95_ms']:.3f} | {row['throughput_megapixels_per_second']:.1f} | "
          f"{mean_speedup_text} | {p95_speedup_text} | "
          f"{row['max_abs_error']:.6g} |")
