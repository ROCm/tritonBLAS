#!/usr/bin/env python3
"""Analyze MALL/L2 counter data from rocprofv3 experiments."""
import csv
import os
import sys
from collections import defaultdict

BASE = os.path.join(os.path.dirname(__file__), "..", "results", "mall_experiments")


def parse_counters(csv_path, kernel_filter="Cijk_"):
    counters = defaultdict(list)
    if not os.path.exists(csv_path):
        return counters
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if kernel_filter not in row.get("Kernel_Name", ""):
                continue
            counters[row["Counter_Name"]].append(float(row["Counter_Value"]))
    return counters


def stats(vals):
    if not vals:
        return {"mean": 0, "min": 0, "max": 0, "count": 0}
    return {
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
        "count": len(vals),
    }


EXPERIMENTS = [
    ("gemm_alone_mall",           "GEMM alone (warm)",       "MALL"),
    ("gemm_alone_l2",             "GEMM alone (warm)",       "L2"),
    ("gemm_rotating_mall",        "GEMM rotating (cold)",    "MALL"),
    ("gemm_rotating_l2",          "GEMM rotating (cold)",    "L2"),
    ("gemm_rccl_mall",            "GEMM+RCCL (warm)",        "MALL"),
    ("gemm_rccl_l2",              "GEMM+RCCL (warm)",        "L2"),
    ("gemm_rccl_rotating_mall",   "GEMM+RCCL rotating",      "MALL"),
    ("gemm_rccl_rotating_l2",     "GEMM+RCCL rotating",      "L2"),
]


def main():
    w = 100
    print("=" * w)
    hdr = f"{'Experiment':<35s} {'Counter':<30s} {'Mean':>14s} {'Min':>14s} {'Max':>14s} {'N':>5s}"
    print(hdr)
    print("=" * w)

    for dirname, label, ctype in EXPERIMENTS:
        csv_path = os.path.join(BASE, dirname, "out_counter_collection.csv")
        counters = parse_counters(csv_path)
        if not counters:
            print(f"{label + ' [' + ctype + ']':<35s}  (no GEMM kernel data)")
            continue
        for cname in sorted(counters.keys()):
            s = stats(counters[cname])
            tag = f"{label} [{ctype}]"
            print(f"{tag:<35s} {cname:<30s} {s['mean']:>14.1f} {s['min']:>14.1f} {s['max']:>14.1f} {s['count']:>5d}")
        print("-" * w)

    # Key analysis
    print("\n" + "=" * 80)
    print("KEY ANALYSIS: L2 Hit Rates")
    print("=" * 80)
    for dirname, label in [
        ("gemm_alone_l2", "GEMM alone (warm)"),
        ("gemm_rotating_l2", "GEMM rotating (cold)"),
        ("gemm_rccl_l2", "GEMM+RCCL (warm)"),
        ("gemm_rccl_rotating_l2", "GEMM+RCCL rotating"),
    ]:
        csv_path = os.path.join(BASE, dirname, "out_counter_collection.csv")
        counters = parse_counters(csv_path)
        hits = counters.get("TCC_HIT_sum", [])
        misses = counters.get("TCC_MISS_sum", [])
        if hits and misses:
            th = sum(hits)
            tm = sum(misses)
            total = th + tm
            rate = th / total * 100 if total > 0 else 0
            print(f"  {label:<30s} L2 hit rate: {rate:6.2f}%  "
                  f"(hits={th:.0f}, misses={tm:.0f})")

    print("\n" + "=" * 80)
    print("KEY ANALYSIS: MALL Bandwidth & HBM Traffic")
    print("=" * 80)
    for dirname, label in [
        ("gemm_alone_mall", "GEMM alone (warm)"),
        ("gemm_rotating_mall", "GEMM rotating (cold)"),
        ("gemm_rccl_mall", "GEMM+RCCL (warm)"),
        ("gemm_rccl_rotating_mall", "GEMM+RCCL rotating"),
    ]:
        csv_path = os.path.join(BASE, dirname, "out_counter_collection.csv")
        counters = parse_counters(csv_path)
        mall = counters.get("MALL_BANDWIDTH_ALL", [])
        hbm_r = counters.get("HBM_READ_BYTES", [])
        hbm_w = counters.get("HBM_WRITE_BYTES", [])
        if mall:
            am = sum(mall) / len(mall)
            ar = sum(hbm_r) / len(hbm_r) if hbm_r else 0
            aw = sum(hbm_w) / len(hbm_w) if hbm_w else 0
            print(f"  {label:<30s} MALL_BW={am:>12.0f}  "
                  f"HBM_RD={ar / 1e6:>8.1f} MB  HBM_WR={aw / 1e6:>8.1f} MB")

    print("\n" + "=" * 80)
    print("KEY ANALYSIS: L2→DRAM Traffic")
    print("=" * 80)
    for dirname, label in [
        ("gemm_alone_l2", "GEMM alone (warm)"),
        ("gemm_rotating_l2", "GEMM rotating (cold)"),
        ("gemm_rccl_l2", "GEMM+RCCL (warm)"),
        ("gemm_rccl_rotating_l2", "GEMM+RCCL rotating"),
    ]:
        csv_path = os.path.join(BASE, dirname, "out_counter_collection.csv")
        counters = parse_counters(csv_path)
        rd = counters.get("TCC_EA0_RDREQ_DRAM_sum", [])
        wr = counters.get("TCC_EA0_WRREQ_DRAM_sum", [])
        if rd and wr:
            ar = sum(rd) / len(rd)
            aw = sum(wr) / len(wr)
            print(f"  {label:<30s} DRAM_RD={ar:>12.0f}  DRAM_WR={aw:>12.0f}  "
                  f"total={ar + aw:>12.0f}")

    # Deltas
    print("\n" + "=" * 80)
    print("DELTAS: RCCL Impact on MALL/HBM")
    print("=" * 80)
    pairs = [
        ("gemm_alone_mall", "gemm_rccl_mall", "warm L2"),
        ("gemm_rotating_mall", "gemm_rccl_rotating_mall", "cold L2 (rotating)"),
    ]
    for base_dir, rccl_dir, desc in pairs:
        b = parse_counters(os.path.join(BASE, base_dir, "out_counter_collection.csv"))
        r = parse_counters(os.path.join(BASE, rccl_dir, "out_counter_collection.csv"))
        for ctr in ["MALL_BANDWIDTH_ALL", "HBM_READ_BYTES", "HBM_WRITE_BYTES"]:
            bv = sum(b[ctr]) / len(b[ctr]) if b[ctr] else 0
            rv = sum(r[ctr]) / len(r[ctr]) if r[ctr] else 0
            delta_pct = (rv - bv) / bv * 100 if bv > 0 else float("inf")
            print(f"  {desc:<25s} {ctr:<25s} base={bv:>12.0f}  rccl={rv:>12.0f}  "
                  f"delta={delta_pct:>+7.1f}%")
        print()

    # MALL hit/miss analysis
    print("=" * 80)
    print("MALL HIT/MISS ANALYSIS (Cache 1)")
    print("=" * 80)
    for dirname, label in [
        ("gemm_alone_mall_hitrate", "GEMM alone"),
        ("gemm_rccl_mall_hitrate", "GEMM+RCCL"),
    ]:
        csv_path = os.path.join(BASE, dirname, "out_counter_collection.csv")
        counters = parse_counters(csv_path)
        total = counters.get("MALL_ALL_REQUESTS_1", [])
        hits = counters.get("MALL_HIT_REQUESTS_1", [])
        misses = counters.get("MALL_MISS_REQUESTS_1", [])
        if total:
            at = sum(total) / len(total)
            ah = sum(hits) / len(hits) if hits else 0
            am = sum(misses) / len(misses) if misses else 0
            rate = ah / at * 100 if at > 0 else 0
            print(f"  {label:<25s} reqs={at:>12.0f}  hits={ah:>12.0f}  "
                  f"misses={am:>12.0f}  hit_rate={rate:5.1f}%")
        else:
            print(f"  {label:<25s} (no data)")

    # GCM local vs remote
    print("\n" + "=" * 80)
    print("GCM LOCAL vs REMOTE TRAFFIC (64B requests)")
    print("=" * 80)
    for dirname, label in [
        ("gemm_alone_gcm", "GEMM alone"),
        ("gemm_rccl_gcm", "GEMM+RCCL"),
    ]:
        csv_path = os.path.join(BASE, dirname, "out_counter_collection.csv")
        counters = parse_counters(csv_path)
        for ctr in ["GCM_READ_LOCAL_UMC_BANDWIDTH",
                     "GCM_READ_REMOTE_UMC_BANDWIDTH",
                     "GCM_WRITE_LOCAL_UMC_BANDWIDTH",
                     "GCM_WRITE_REMOTE_UMC_BANDWIDTH"]:
            vals = counters.get(ctr, [])
            if vals:
                avg = sum(vals) / len(vals)
                short = ctr.replace("GCM_", "").replace("_BANDWIDTH", "")
                print(f"  {label:<20s} {short:<30s} {avg:>14.0f}")
        if not any(counters.get(c) for c in ["GCM_READ_LOCAL_UMC_BANDWIDTH"]):
            print(f"  {label:<20s} (no data)")
        print()

    # TCC credit stalls
    print("=" * 80)
    print("TCC CREDIT STALL ANALYSIS")
    print("=" * 80)
    for dirname, label in [
        ("gemm_alone_stalls", "GEMM alone"),
        ("gemm_rccl_stalls", "GEMM+RCCL"),
    ]:
        csv_path = os.path.join(BASE, dirname, "out_counter_collection.csv")
        counters = parse_counters(csv_path)
        for ctr in ["TCC_EA0_RDREQ_DRAM_CREDIT_STALL_sum",
                     "TCC_EA0_WRREQ_DRAM_CREDIT_STALL_sum",
                     "TCC_EA0_RDREQ_GMI_CREDIT_STALL_sum",
                     "TCC_EA0_WRREQ_GMI_CREDIT_STALL_sum"]:
            vals = counters.get(ctr, [])
            if vals:
                avg = sum(vals) / len(vals)
                short = ctr.replace("TCC_EA0_", "").replace("_sum", "")
                print(f"  {label:<20s} {short:<35s} {avg:>14.0f}")
        if not any(counters.get(c) for c in ["TCC_EA0_RDREQ_DRAM_CREDIT_STALL_sum"]):
            print(f"  {label:<20s} (no data)")
        print()


if __name__ == "__main__":
    main()
