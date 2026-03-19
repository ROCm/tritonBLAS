#!/usr/bin/env python3
"""
Unified rocprofv3 counter parser for overlap analysis.

Handles both standard L2 counters (TCC_*) and df-counter MALL/HBM counters.
Produces clean JSON/CSV summaries suitable for the autoresearch experiment log.

Usage:
    # Parse a single rocprofv3 CSV output
    python3 benchmarks/parse_counters.py /path/to/out_counter_collection.csv

    # Parse and compare baseline vs treatment
    python3 benchmarks/parse_counters.py --baseline results/l2_gemm_alone/ --treatment results/l2_gemm_rccl/

    # Output as JSON for experiment log
    python3 benchmarks/parse_counters.py --json /path/to/csv
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def parse_rocprof_csv(csv_path):
    """Parse a rocprofv3 counter collection CSV.

    Returns a list of dicts, one per kernel dispatch, with counter values.
    Handles the rocprofv3 format where each counter gets its own row.
    """
    dispatches = defaultdict(lambda: {"counters": {}})

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            disp_id = row.get("Dispatch_Id", row.get("dispatch_id", ""))
            if not disp_id:
                continue

            kernel = row.get("Kernel_Name", row.get("kernel_name", ""))
            counter = row.get("Counter_Name", row.get("counter_name", ""))
            value = row.get("Counter_Value", row.get("counter_value", "0"))

            key = f"{disp_id}"
            dispatches[key]["dispatch_id"] = disp_id
            dispatches[key]["kernel_name"] = kernel
            dispatches[key]["grid_size"] = row.get("Grid_Size", "")
            dispatches[key]["workgroup_size"] = row.get("Workgroup_Size", "")

            try:
                dispatches[key]["counters"][counter] = float(value)
            except (ValueError, TypeError):
                dispatches[key]["counters"][counter] = value

    return list(dispatches.values())


def filter_gemm_dispatches(dispatches):
    """Filter to only GEMM kernel dispatches (Cijk_ or matmul-related)."""
    gemm_keywords = ["Cijk_", "matmul", "gemm", "ws_matmul", "persistent_matmul"]
    result = []
    for d in dispatches:
        name = d.get("kernel_name", "").lower()
        if any(kw.lower() in name for kw in gemm_keywords):
            result.append(d)
    return result


def compute_l2_stats(dispatches):
    """Compute L2 cache statistics from TCC counters."""
    total_hit = 0
    total_miss = 0
    total_read = 0
    total_write = 0
    total_writeback = 0
    n = 0

    for d in dispatches:
        c = d["counters"]
        hit = c.get("TCC_HIT_sum", c.get("TCC_HIT", 0))
        miss = c.get("TCC_MISS_sum", c.get("TCC_MISS", 0))
        read = c.get("TCC_READ_sum", c.get("TCC_READ", 0))
        write = c.get("TCC_WRITE_sum", c.get("TCC_WRITE", 0))
        wb = c.get("TCC_WRITEBACK_sum", c.get("TCC_WRITEBACK", 0))

        if isinstance(hit, (int, float)) and isinstance(miss, (int, float)):
            total_hit += hit
            total_miss += miss
            total_read += read
            total_write += write
            total_writeback += wb
            n += 1

    total = total_hit + total_miss
    hit_rate = (total_hit / total * 100) if total > 0 else 0.0

    return {
        "n_dispatches": n,
        "total_hit": total_hit,
        "total_miss": total_miss,
        "hit_rate_pct": round(hit_rate, 2),
        "total_read": total_read,
        "total_write": total_write,
        "total_writeback": total_writeback,
    }


def compute_mall_stats(dispatches):
    """Compute MALL/LLC and HBM statistics from df-counter output."""
    total_mall_bw = 0
    total_hbm_read = 0
    total_hbm_write = 0
    n = 0

    for d in dispatches:
        c = d["counters"]
        mall = c.get("MALL_BANDWIDTH_ALL", 0)
        hbm_r = c.get("HBM_READ_BYTES", 0)
        hbm_w = c.get("HBM_WRITE_BYTES", 0)

        if isinstance(mall, (int, float)):
            total_mall_bw += mall
            total_hbm_read += hbm_r
            total_hbm_write += hbm_w
            n += 1

    return {
        "n_dispatches": n,
        "total_mall_bandwidth": total_mall_bw,
        "total_hbm_read_bytes": total_hbm_read,
        "total_hbm_write_bytes": total_hbm_write,
        "total_hbm_bytes": total_hbm_read + total_hbm_write,
    }


def summarize_csv(csv_path, gemm_only=True):
    """Parse a CSV and return a full summary."""
    dispatches = parse_rocprof_csv(csv_path)
    if gemm_only:
        gemm = filter_gemm_dispatches(dispatches)
    else:
        gemm = dispatches

    summary = {
        "file": str(csv_path),
        "total_dispatches": len(dispatches),
        "gemm_dispatches": len(gemm),
    }

    has_tcc = any("TCC_HIT" in d["counters"] or "TCC_HIT_sum" in d["counters"] for d in gemm)
    has_mall = any("MALL_BANDWIDTH_ALL" in d["counters"] for d in gemm)

    if has_tcc:
        summary["l2"] = compute_l2_stats(gemm)
    if has_mall:
        summary["mall"] = compute_mall_stats(gemm)

    return summary


def compare_results(baseline_dir, treatment_dir, gemm_only=True):
    """Compare baseline and treatment counter data."""
    def find_csvs(d):
        p = Path(d)
        return sorted(p.rglob("*counter_collection*.csv"))

    baseline_csvs = find_csvs(baseline_dir)
    treatment_csvs = find_csvs(treatment_dir)

    comparison = {
        "baseline_dir": str(baseline_dir),
        "treatment_dir": str(treatment_dir),
        "baseline_files": len(baseline_csvs),
        "treatment_files": len(treatment_csvs),
    }

    if baseline_csvs:
        b_summary = summarize_csv(baseline_csvs[0], gemm_only)
        comparison["baseline"] = b_summary

    if treatment_csvs:
        t_summary = summarize_csv(treatment_csvs[0], gemm_only)
        comparison["treatment"] = t_summary

    if "baseline" in comparison and "treatment" in comparison:
        b = comparison["baseline"]
        t = comparison["treatment"]
        diff = {}
        if "l2" in b and "l2" in t:
            diff["l2_hit_rate_baseline"] = b["l2"]["hit_rate_pct"]
            diff["l2_hit_rate_treatment"] = t["l2"]["hit_rate_pct"]
            diff["l2_hit_rate_delta"] = round(t["l2"]["hit_rate_pct"] - b["l2"]["hit_rate_pct"], 2)
        if "mall" in b and "mall" in t:
            diff["mall_bw_baseline"] = b["mall"]["total_mall_bandwidth"]
            diff["mall_bw_treatment"] = t["mall"]["total_mall_bandwidth"]
            b_hbm = b["mall"]["total_hbm_bytes"]
            t_hbm = t["mall"]["total_hbm_bytes"]
            diff["hbm_increase_pct"] = round((t_hbm - b_hbm) / b_hbm * 100, 2) if b_hbm > 0 else None
        comparison["diff"] = diff

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Parse rocprofv3 counter data")
    parser.add_argument("csv_file", nargs="?", help="Single CSV file to parse")
    parser.add_argument("--baseline", help="Baseline results directory")
    parser.add_argument("--treatment", help="Treatment results directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--all-kernels", action="store_true", help="Include non-GEMM kernels")
    args = parser.parse_args()

    gemm_only = not args.all_kernels

    if args.baseline and args.treatment:
        result = compare_results(args.baseline, args.treatment, gemm_only)
    elif args.csv_file:
        result = summarize_csv(args.csv_file, gemm_only)
    else:
        parser.print_help()
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if "diff" in result:
            print(f"\n{'='*60}")
            print("COUNTER COMPARISON")
            print(f"{'='*60}")
            print(f"Baseline:  {result['baseline_dir']}")
            print(f"Treatment: {result['treatment_dir']}")
            d = result["diff"]
            if "l2_hit_rate_baseline" in d:
                print(f"\nL2 Cache Hit Rate:")
                print(f"  Baseline:  {d['l2_hit_rate_baseline']:.1f}%")
                print(f"  Treatment: {d['l2_hit_rate_treatment']:.1f}%")
                print(f"  Delta:     {d['l2_hit_rate_delta']:+.1f}%")
            if "hbm_increase_pct" in d:
                print(f"\nHBM Traffic Increase: {d['hbm_increase_pct']:+.1f}%")
                print(f"MALL BW Baseline:     {d['mall_bw_baseline']}")
                print(f"MALL BW Treatment:    {d['mall_bw_treatment']}")
        else:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
