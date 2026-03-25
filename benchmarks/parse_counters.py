#!/usr/bin/env python3
"""Parse rocprofv3 counter results (SQLite for alone, CSV for RCCL) for 8K GEMM."""
import csv
import glob
import json
import os
import sqlite3
import statistics
from collections import defaultdict

OUTDIR = "results/counters_8k"

GEMM_PATTERNS = ["Cijk_", "ws_hierarchical", "matmul_kernel"]
SKIP_PATTERNS = ["distribution", "elementwise", "fill", "copy", "reduce",
                 "nccl", "rccl", "AllReduce", "ncclKernel", "ncclDevKernel",
                 "rocclr_fill", "Sendrecv"]


def is_gemm_kernel(name):
    nl = name.lower()
    for skip in SKIP_PATTERNS:
        if skip.lower() in nl:
            return False
    for pat in GEMM_PATTERNS:
        if pat.lower() in nl:
            return True
    return False


def parse_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    tables = [t[0] for t in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    pmc_tbl = [t for t in tables if "pmc_event" in t][0]
    info_tbl = [t for t in tables if "info_pmc" in t][0]
    kd_tbl = [t for t in tables if "kernel_dispatch" in t][0]
    ks_tbl = [t for t in tables if "kernel_symbol" in t][0]

    counter_names = {}
    for row in conn.execute(f"SELECT id, name FROM {info_tbl}"):
        counter_names[row[0]] = row[1]

    gemm_event_ids = set()
    for row in conn.execute(f"""
        SELECT d.event_id, s.kernel_name
        FROM {kd_tbl} d JOIN {ks_tbl} s ON d.kernel_id = s.id
    """):
        if is_gemm_kernel(row[1]):
            gemm_event_ids.add(row[0])

    results = defaultdict(list)
    for row in conn.execute(f"SELECT event_id, pmc_id, value FROM {pmc_tbl}"):
        event_id, pmc_id, value = row
        if event_id in gemm_event_ids:
            cname = counter_names.get(pmc_id, str(pmc_id))
            results[cname].append(value)

    conn.close()
    return {k: {"median": statistics.median(v), "mean": statistics.mean(v),
                "count": len(v)} for k, v in results.items() if v}


def parse_csv_dir(dir_path):
    csv_files = sorted(glob.glob(os.path.join(dir_path, "*", "*_counter_collection.csv")))
    if not csv_files:
        return {}

    pids = set()
    for f in csv_files:
        base = os.path.basename(f).split("_")[0]
        pids.add(base)
    first_pid = sorted(pids)[0]
    rank0_files = [f for f in csv_files if f"/{first_pid}_" in f]

    results = defaultdict(list)
    for csv_file in rank0_files:
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kname = row.get("Kernel_Name", "")
                ctr_name = row.get("Counter_Name", "")
                ctr_val = row.get("Counter_Value", "")
                if not ctr_name or not ctr_val:
                    continue
                if is_gemm_kernel(kname):
                    try:
                        results[ctr_name].append(float(ctr_val))
                    except ValueError:
                        pass

    return {k: {"median": statistics.median(v), "mean": statistics.mean(v),
                "count": len(v)} for k, v in results.items() if v}


def main():
    scenarios = {}
    for backend in ["torch", "ws"]:
        for mode in ["alone", "rccl"]:
            for pass_name in ["l2", "mall", "hbm"]:
                tag = f"{backend}_{mode}_{pass_name}"
                db_path = os.path.join(OUTDIR, f"{tag}_results.db")
                dir_path = os.path.join(OUTDIR, f"{tag}_dir")

                if os.path.exists(db_path):
                    data = parse_sqlite(db_path)
                elif os.path.isdir(dir_path):
                    data = parse_csv_dir(dir_path)
                else:
                    continue

                key = f"{backend}_{mode}"
                if key not in scenarios:
                    scenarios[key] = {}
                scenarios[key].update(data)

    def fmt(v):
        if v != v: return "—"
        if abs(v) >= 1e9: return f"{v:.2e}"
        if abs(v) >= 1e6: return f"{v/1e6:.1f}M"
        if abs(v) >= 1e3: return f"{v/1e3:.1f}K"
        return f"{v:.2f}"

    def delta(a, b):
        if a != a or b != b or a == 0: return "—"
        return f"{(b - a) / a * 100:+.1f}%"

    print("\n" + "=" * 120)
    print(f"{'Counter':40s} │ {'torch alone':>12s} │ {'torch+RCCL':>12s} │ {'Δ':>8s} │ "
          f"{'WS alone':>12s} │ {'WS+RCCL':>12s} │ {'Δ':>8s}")
    print("=" * 120)

    all_counters = sorted(set().union(*(s.keys() for s in scenarios.values())))
    groups = {"L2": [], "MALL": [], "HBM": [], "Other": []}
    for c in all_counters:
        if "TCC" in c: groups["L2"].append(c)
        elif "MALL" in c: groups["MALL"].append(c)
        elif "HBM" in c: groups["HBM"].append(c)
        else: groups["Other"].append(c)

    for group_name, ctrs in groups.items():
        if not ctrs: continue
        print(f"\n  --- {group_name} ---")
        for ctr in ctrs:
            ta = scenarios.get("torch_alone", {}).get(ctr, {}).get("median", float("nan"))
            tr = scenarios.get("torch_rccl", {}).get(ctr, {}).get("median", float("nan"))
            wa = scenarios.get("ws_alone", {}).get(ctr, {}).get("median", float("nan"))
            wr = scenarios.get("ws_rccl", {}).get(ctr, {}).get("median", float("nan"))
            print(f"  {ctr:38s} │ {fmt(ta):>12s} │ {fmt(tr):>12s} │ {delta(ta,tr):>8s} │ "
                  f"{fmt(wa):>12s} │ {fmt(wr):>12s} │ {delta(wa,wr):>8s}")

    with open(os.path.join(OUTDIR, "counter_summary.json"), "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"\nJSON: {OUTDIR}/counter_summary.json")


if __name__ == "__main__":
    main()
