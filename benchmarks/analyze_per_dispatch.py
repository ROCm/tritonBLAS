#!/usr/bin/env python3
"""Per-dispatch analysis of WS kernel counters and timestamps.

Looks for patterns in per-iteration timing and counters:
- Is every Nth dispatch faster/slower? (rotating buffer pattern)
- Do early dispatches differ from later ones? (warm-up effect)
- What's the inter-dispatch gap? (pipelining evidence)
"""
import csv
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "ws_cache"


def parse_dispatches(csv_path, kernel_filter="ws_persistent"):
    """Parse rocprofv3 CSV into per-dispatch dicts."""
    dispatches = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if kernel_filter not in row.get("Kernel_Name", ""):
                continue
            did = int(row["Dispatch_Id"])
            cn = row["Counter_Name"]
            cv = float(row["Counter_Value"])
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            if did not in dispatches:
                dispatches[did] = {"start": start, "end": end, "counters": {}}
            dispatches[did]["counters"][cn] = cv
    return dispatches


def analyze(label, csv_path):
    dispatches = parse_dispatches(csv_path)
    if not dispatches:
        print(f"\n  {label}: NO DATA")
        return

    sorted_ids = sorted(dispatches.keys())

    sep = "=" * 95
    print(f"\n{sep}")
    print(f"  {label} ({len(sorted_ids)} dispatches)")
    print(f"{sep}")

    header = f"{'#':>3} {'Disp':>5} {'Dur(ms)':>9} {'TCC_HIT':>12} {'TCC_MISS':>12} {'HitRate':>8} {'Gap(ms)':>9}"
    print(header)
    print("-" * 95)

    prev_end = None
    durs = []
    hits_list = []
    misses_list = []
    gaps = []

    for i, did in enumerate(sorted_ids):
        d = dispatches[did]
        dur = (d["end"] - d["start"]) / 1e6
        hit = d["counters"].get("TCC_HIT_sum", 0)
        miss = d["counters"].get("TCC_MISS_sum", 0)
        total = hit + miss
        rate = hit / total * 100 if total else 0
        gap = (d["start"] - prev_end) / 1e6 if prev_end else 0.0
        prev_end = d["end"]

        buf_idx = ""
        if "rotating" in label.lower():
            buf_idx = f" [buf {i % 4}]"

        print(f"{i:>3} {did:>5} {dur:>9.3f} {hit:>12,.0f} {miss:>12,.0f} {rate:>7.2f}% {gap:>9.3f}{buf_idx}")

        durs.append(dur)
        hits_list.append(hit)
        misses_list.append(miss)
        if i > 0:
            gaps.append(gap)

    n = len(durs)
    mean_dur = sum(durs) / n
    stdev_dur = (sum((x - mean_dur) ** 2 for x in durs) / n) ** 0.5

    print(f"\nDuration: mean={mean_dur:.3f}  stdev={stdev_dur:.3f}  min={min(durs):.3f}  max={max(durs):.3f}")
    print(f"TCC_HIT:  mean={sum(hits_list)/n:,.0f}  min={min(hits_list):,.0f}  max={max(hits_list):,.0f}")
    print(f"TCC_MISS: mean={sum(misses_list)/n:,.0f}  min={min(misses_list):,.0f}  max={max(misses_list):,.0f}")
    if gaps:
        print(f"Gap:      mean={sum(gaps)/len(gaps):.3f}  min={min(gaps):.3f}  max={max(gaps):.3f}")

    if "rotating" in label.lower() and n >= 8:
        print(f"\nPer-buffer-index analysis (4 rotating buffers):")
        for buf in range(4):
            buf_durs = [durs[i] for i in range(n) if i % 4 == buf]
            buf_hits = [hits_list[i] for i in range(n) if i % 4 == buf]
            buf_misses = [misses_list[i] for i in range(n) if i % 4 == buf]
            if buf_durs:
                bh = sum(buf_hits) / len(buf_hits)
                bm = sum(buf_misses) / len(buf_misses)
                bt = bh + bm
                rate = bh / bt * 100 if bt else 0
                print(f"  buf[{buf}]: dur={sum(buf_durs)/len(buf_durs):.3f}ms  "
                      f"hit_rate={rate:.2f}%  miss={bm:,.0f}")


def main():
    l2_dir = RESULTS_DIR / "l2_hitmiss"

    conditions = [
        ("ALONE WARM", l2_dir / "alone_warm__l2_hitmiss"),
        ("ALONE ROTATING", l2_dir / "alone_rotating__l2_hitmiss"),
        ("RCCL WARM", l2_dir / "rccl_warm__l2_hitmiss"),
        ("RCCL ROTATING", l2_dir / "rccl_rotating__l2_hitmiss"),
    ]

    for label, cond_dir in conditions:
        csv_path = cond_dir / "out_counter_collection.csv"
        if csv_path.exists():
            analyze(label, str(csv_path))
        else:
            print(f"\n  {label}: file not found ({csv_path})")


if __name__ == "__main__":
    main()
