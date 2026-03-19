#!/usr/bin/env python3
"""Parse rocprofv3 kernel_trace CSV and show RCCL + GEMM timeline during overlap."""
import csv
import sys


def short_name(kname):
    if "ws_persistent_matmul" in kname:
        return "WS_GEMM"
    if "persistent_matmul" in kname:
        return "PERSISTENT_GEMM"
    if "ncclDevKernel" in kname or "nccl" in kname.lower():
        return "NCCL_KERNEL"
    if "spin_kernel" in kname:
        return "GPU_SLEEP"
    if "vectorized_elementwise" in kname:
        return "FILL/ZERO"
    if "distribution_elementwise" in kname:
        return "RANDN"
    if "Cijk" in kname or "hipblas" in kname.lower():
        return "HIPBLAS_GEMM"
    return kname[:50]


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_rccl_trace.py <kernel_trace.csv>")
        sys.exit(1)

    path = sys.argv[1]

    dispatches = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            grid_x = int(row["Grid_Size_X"])
            wg_x = int(row["Workgroup_Size_X"])
            dispatches.append({
                "name": short_name(row["Kernel_Name"]),
                "full_name": row["Kernel_Name"][:100],
                "start": int(row["Start_Timestamp"]),
                "end": int(row["End_Timestamp"]),
                "wgs": grid_x // wg_x if wg_x > 0 else grid_x,
                "queue": row["Queue_Id"],
                "stream": row["Stream_Id"],
            })

    if not dispatches:
        print("No dispatches found")
        return

    # Find the overlap region: look for the pattern GPU_SLEEP → WS_GEMM that follows NCCL
    # First, find all GEMM and NCCL dispatches
    gemm_dispatches = [d for d in dispatches if "GEMM" in d["name"]]
    nccl_dispatches = [d for d in dispatches if "NCCL" in d["name"]]
    sleep_dispatches = [d for d in dispatches if "SLEEP" in d["name"]]

    print(f"Total dispatches: {len(dispatches)}")
    print(f"  GEMM: {len(gemm_dispatches)}")
    print(f"  NCCL: {len(nccl_dispatches)}")
    print(f"  Sleep: {len(sleep_dispatches)}")

    if not gemm_dispatches or not nccl_dispatches:
        print("No GEMM or NCCL dispatches found")
        return

    # Find overlapped iterations: look for NCCL + GEMM that are temporally close
    # Sort all interesting dispatches by start time
    interesting = [d for d in dispatches if d["name"] in ("WS_GEMM", "PERSISTENT_GEMM", "NCCL_KERNEL", "GPU_SLEEP", "FILL/ZERO", "HIPBLAS_GEMM")]
    interesting.sort(key=lambda x: x["start"])

    # Find the last N GEMM dispatches (these are the timed iterations)
    # The overlap phase has: [NCCL, SLEEP, GEMM] repeated
    t0 = interesting[0]["start"] if interesting else 0

    print(f"\n{'Kernel':<20s} {'Q':>2s} {'S':>2s} {'WGs':>5s} "
          f"{'Start(us)':>12s} {'End(us)':>12s} {'Dur(us)':>10s}")
    print("-" * 75)

    # Show only the last portion (timed overlap + alone)
    # Let's show everything after the first NCCL kernel that's near a GEMM
    # Find pattern: look for NCCL that starts within 1ms of a GEMM
    overlap_start = None
    for i, d in enumerate(interesting):
        if d["name"] == "NCCL_KERNEL":
            # Check if there's a GEMM within 2ms after
            for j in range(i+1, min(i+5, len(interesting))):
                if "GEMM" in interesting[j]["name"]:
                    overlap_start = d["start"]
                    break
        if overlap_start:
            break

    if not overlap_start:
        # Fall back: show last 30 dispatches
        interesting = interesting[-30:]
        t0 = interesting[0]["start"]
    else:
        # Show from 10ms before first overlap
        interesting = [d for d in interesting if d["start"] >= overlap_start - 10_000_000]
        t0 = interesting[0]["start"]

    for d in interesting:
        s = (d["start"] - t0) / 1000
        e = (d["end"] - t0) / 1000
        dur = (d["end"] - d["start"]) / 1000
        print(f"{d['name']:<20s} {d['queue']:>2s} {d['stream']:>2s} {d['wgs']:>5d} "
              f"{s:>12.1f} {e:>12.1f} {dur:>10.1f}")

    # Analyze temporal overlap between NCCL and GEMM in timed iterations
    print("\n--- Overlap Analysis ---")
    # Find overlapping pairs
    for gd in gemm_dispatches[-6:]:  # Last 6 GEMMs (3 alone + 3 overlap)
        g_start, g_end = gd["start"], gd["end"]
        g_dur = (g_end - g_start) / 1000

        # Find NCCL kernels that overlap with this GEMM
        overlapping_nccl = []
        for nd in nccl_dispatches:
            # Check temporal overlap
            if nd["start"] < g_end and nd["end"] > g_start:
                overlap_start_t = max(nd["start"], g_start)
                overlap_end_t = min(nd["end"], g_end)
                overlap_dur = (overlap_end_t - overlap_start_t) / 1000
                overlapping_nccl.append((nd, overlap_dur))

        if overlapping_nccl:
            total_nccl_overlap = sum(d for _, d in overlapping_nccl)
            nccl_dur = (overlapping_nccl[0][0]["end"] - overlapping_nccl[0][0]["start"]) / 1000
            print(f"\n  GEMM: {g_dur:.1f}us (Q{gd['queue']},S{gd['stream']})")
            print(f"    Overlapping NCCL: {len(overlapping_nccl)} kernel(s), "
                  f"overlap duration: {total_nccl_overlap:.1f}us")
            print(f"    NCCL kernel duration: {nccl_dur:.1f}us")
            print(f"    NCCL starts: {(overlapping_nccl[0][0]['start'] - g_start)/1000:.1f}us before/after GEMM start")
        else:
            print(f"\n  GEMM: {g_dur:.1f}us (Q{gd['queue']},S{gd['stream']}) - NO NCCL overlap")


if __name__ == "__main__":
    main()
