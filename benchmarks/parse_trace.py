#!/usr/bin/env python3
"""Parse rocprofv3 kernel_trace CSV and show GPU timeline for GEMM dispatches."""
import csv
import sys


def short_name(kname):
    if "ws_persistent_matmul" in kname:
        return "ws_persistent_matmul"
    if "persistent_matmul" in kname:
        return "persistent_matmul"
    if "spin_kernel" in kname:
        return "gpu_sleep"
    if "cu_hog" in kname:
        return "cu_hog"
    if "vectorized_elementwise" in kname:
        if "BFloat16" in kname and "2ul" in kname:
            return "add_(pollution)"
        return "fill_/zero_"
    if "distribution_elementwise" in kname:
        return "randn_init"
    return kname[:60]


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_trace.py <kernel_trace.csv> [label]")
        sys.exit(1)

    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path

    dispatches = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            grid_x = int(row["Grid_Size_X"])
            wg_x = int(row["Workgroup_Size_X"])
            dispatches.append({
                "name": short_name(row["Kernel_Name"]),
                "start": int(row["Start_Timestamp"]),
                "end": int(row["End_Timestamp"]),
                "grid_x": grid_x,
                "wg_x": wg_x,
                "wgs": grid_x // wg_x if wg_x > 0 else grid_x,
                "queue": row["Queue_Id"],
                "stream": row["Stream_Id"],
            })

    if not dispatches:
        print("No dispatches found")
        return

    t0 = min(d["start"] for d in dispatches)

    print(f"\n=== {label} ===")
    print(f"{'Kernel':<25s} {'Q':>2s} {'S':>2s} {'WGs':>6s} "
          f"{'Start(us)':>10s} {'End(us)':>10s} {'Dur(us)':>10s}")
    print("-" * 70)

    # Show only timed dispatches (skip warmup: find last GEMM block)
    gemm_names = {"ws_persistent_matmul", "persistent_matmul"}
    sorted_d = sorted(dispatches, key=lambda x: x["start"])

    # Find the start of the timed iterations (after warmup gap)
    gemm_indices = [i for i, d in enumerate(sorted_d) if d["name"] in gemm_names]
    if len(gemm_indices) >= 6:
        # Warmup = first 3, timed = last 3
        timed_start = sorted_d[gemm_indices[3]]["start"] - 100_000_000  # 100ms before
    else:
        timed_start = 0

    for d in sorted_d:
        if d["start"] < timed_start:
            continue
        if d["name"] == "randn_init":
            continue
        s = (d["start"] - t0) / 1000
        e = (d["end"] - t0) / 1000
        dur = (d["end"] - d["start"]) / 1000
        print(f"{d['name']:<25s} {d['queue']:>2s} {d['stream']:>2s} {d['wgs']:>6d} "
              f"{s:>10.1f} {e:>10.1f} {dur:>10.1f}")

    # Summary
    gemm_dispatches = [d for d in sorted_d if d["name"] in gemm_names and d["start"] >= timed_start]
    if gemm_dispatches:
        durs = [(d["end"] - d["start"]) / 1000 for d in gemm_dispatches]
        print(f"\nGEMM dispatches: {len(durs)}, "
              f"durations: {', '.join(f'{d:.1f}' for d in durs)} us")


if __name__ == "__main__":
    main()
