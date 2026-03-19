#!/usr/bin/env python3
"""Parse rocprofv3 counter_collection CSV and summarize TCC/TCP stats for GEMM kernels."""
import csv
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_l2_counters.py <counter_collection.csv>")
        sys.exit(1)

    path = sys.argv[1]
    # Track (kernel_name, dispatch_id) pairs and counter values
    kernel_counters = defaultdict(lambda: defaultdict(float))
    kernel_dispatches = defaultdict(set)

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            kname = row["Kernel_Name"]
            dispatch_id = row["Dispatch_Id"]
            counter = row["Counter_Name"]
            value = float(row["Counter_Value"])
            kernel_counters[kname][counter] += value
            kernel_dispatches[kname].add(dispatch_id)

    # Sort by total counter activity
    def sort_key(item):
        return -sum(item[1].values())

    for kname, counters in sorted(kernel_counters.items(), key=sort_key):
        dispatches = len(kernel_dispatches[kname])
        if dispatches == 0:
            continue

        total_activity = sum(counters.values())
        if total_activity < 10000:
            continue

        short = kname if len(kname) <= 80 else kname[:40] + " ... " + kname[-35:]

        print(f"Kernel: {short}")
        print(f"  Dispatches:   {dispatches}")
        for cname, cval in sorted(counters.items()):
            per_d = cval / dispatches
            print(f"  {cname:30s} {cval:>15,.0f}  ({per_d:>12,.0f} /dispatch)")
        if "TCC_HIT" in counters and "TCC_MISS" in counters:
            hit, miss = counters["TCC_HIT"], counters["TCC_MISS"]
            total = hit + miss
            if total > 0:
                print(f"  L2 Hit Rate:  {hit/total*100:.1f}%")
        print()


if __name__ == "__main__":
    main()
