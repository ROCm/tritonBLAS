#!/usr/bin/env python3
"""Comprehensive parser for v2 counter collection: MALL, HBM, L2 with per-kernel-type breakdown."""
import csv, glob, json, os, sqlite3, statistics
from collections import defaultdict

OUTDIR = "results/counters_8k_v2"


def classify_kernel(name):
    nl = name.lower()
    if "cijk_" in nl: return "GEMM"
    if "ws_hierarchical" in nl: return "GEMM"
    if any(k in nl for k in ["nccl", "rccl", "allreduce", "sendrecv", "ncclkernel", "nccldev"]): return "RCCL"
    if "distribution" in nl or "normal" in nl: return "randn"
    if "fill" in nl: return "fill"
    if "copy" in nl: return "copy"
    return "other"


def parse_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    pmc_tbl = [t for t in tables if "pmc_event" in t][0]
    info_tbl = [t for t in tables if "info_pmc" in t][0]
    kd_tbl = [t for t in tables if "kernel_dispatch" in t][0]
    ks_tbl = [t for t in tables if "kernel_symbol" in t][0]

    ctr_names = dict(conn.execute(f"SELECT id, name FROM {info_tbl}").fetchall())
    event_kernel = {}
    for row in conn.execute(f"SELECT d.event_id, s.kernel_name FROM {kd_tbl} d JOIN {ks_tbl} s ON d.kernel_id = s.id"):
        event_kernel[row[0]] = classify_kernel(row[1])

    by_type = defaultdict(lambda: defaultdict(list))
    for row in conn.execute(f"SELECT event_id, pmc_id, value FROM {pmc_tbl}"):
        eid, pid, val = row
        ktype = event_kernel.get(eid, "other")
        cname = ctr_names.get(pid, str(pid))
        by_type[ktype][cname].append(val)

    conn.close()
    result = {}
    for ktype, ctrs in by_type.items():
        result[ktype] = {}
        for c, vals in ctrs.items():
            result[ktype][c] = {"median": statistics.median(vals), "mean": statistics.mean(vals), "n": len(vals)}
    return result


def parse_csv_dir(dir_path):
    csv_files = sorted(glob.glob(os.path.join(dir_path, "*", "*_counter_collection.csv")))
    if not csv_files:
        return {}
    pids = sorted(set(os.path.basename(f).split("_")[0] for f in csv_files))
    first_pid = pids[0]
    rank0 = [f for f in csv_files if f"/{first_pid}_" in f]

    by_type = defaultdict(lambda: defaultdict(list))
    for csv_file in rank0:
        with open(csv_file, newline="") as f:
            for row in csv.DictReader(f):
                ktype = classify_kernel(row.get("Kernel_Name", ""))
                cn = row.get("Counter_Name", "")
                cv = row.get("Counter_Value", "")
                if cn and cv:
                    try:
                        by_type[ktype][cn].append(float(cv))
                    except ValueError:
                        pass

    result = {}
    for ktype, ctrs in by_type.items():
        result[ktype] = {}
        for c, vals in ctrs.items():
            result[ktype][c] = {"median": statistics.median(vals), "mean": statistics.mean(vals), "n": len(vals)}
    return result


def main():
    scenarios = {}
    for backend in ["torch", "ws"]:
        for mode in ["alone", "rccl"]:
            key = f"{backend}_{mode}"
            merged = {}
            for pn in ["p1", "p2", "p3", "p4", "p5"]:
                tag = f"{key}_{pn}"
                db = os.path.join(OUTDIR, f"{tag}_results.db")
                d = os.path.join(OUTDIR, f"{tag}_dir")
                if os.path.exists(db):
                    data = parse_sqlite(db)
                elif os.path.isdir(d):
                    data = parse_csv_dir(d)
                else:
                    continue
                for ktype, ctrs in data.items():
                    if ktype not in merged:
                        merged[ktype] = {}
                    merged[ktype].update(ctrs)
            scenarios[key] = merged

    def fmt(v):
        if v != v: return "—"
        if abs(v) >= 1e9: return f"{v/1e9:.2f}G"
        if abs(v) >= 1e6: return f"{v/1e6:.1f}M"
        if abs(v) >= 1e3: return f"{v/1e3:.1f}K"
        if abs(v) < 1: return f"{v:.4f}"
        return f"{v:.1f}"

    def delta(a, b):
        if a != a or b != b or a == 0: return "—"
        return f"{(b - a) / a * 100:+.1f}%"

    def print_section(title, counters, kernel_type):
        print(f"\n{'─'*120}")
        print(f"  {title}  (kernel type: {kernel_type})")
        print(f"{'─'*120}")
        print(f"  {'Counter':38s} │ {'torch alone':>12s} │ {'torch+RCCL':>12s} │ {'Δ':>8s} │ "
              f"{'WS alone':>12s} │ {'WS+RCCL':>12s} │ {'Δ':>8s}")
        print(f"  {'─'*38}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*8}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*8}")
        for ctr in counters:
            ta = scenarios.get("torch_alone", {}).get(kernel_type, {}).get(ctr, {}).get("median", float("nan"))
            tr = scenarios.get("torch_rccl", {}).get(kernel_type, {}).get(ctr, {}).get("median", float("nan"))
            wa = scenarios.get("ws_alone", {}).get(kernel_type, {}).get(ctr, {}).get("median", float("nan"))
            wr = scenarios.get("ws_rccl", {}).get(kernel_type, {}).get(ctr, {}).get("median", float("nan"))
            print(f"  {ctr:38s} │ {fmt(ta):>12s} │ {fmt(tr):>12s} │ {delta(ta,tr):>8s} │ "
                  f"{fmt(wa):>12s} │ {fmt(wr):>12s} │ {delta(wa,wr):>8s}")

    # Cross-validate hit rate from raw counts
    print("\n" + "=" * 120)
    print("  CROSS-VALIDATION: Compute hit rate from raw MALL_HIT_REQUESTS / MALL_ALL_REQUESTS")
    print("=" * 120)
    for key in ["torch_alone", "torch_rccl", "ws_alone", "ws_rccl"]:
        gemm = scenarios.get(key, {}).get("GEMM", {})
        rccl = scenarios.get(key, {}).get("RCCL", {})
        for ktype, label, data in [("GEMM", "GEMM", gemm), ("RCCL", "RCCL", rccl)]:
            hit1 = data.get("MALL_HIT_REQUESTS_1", {}).get("median", 0)
            all1 = data.get("MALL_ALL_REQUESTS_1", {}).get("median", 0)
            hit2 = data.get("MALL_HIT_REQUESTS_2", {}).get("median", 0)
            all2 = data.get("MALL_ALL_REQUESTS_2", {}).get("median", 0)
            hr1 = data.get("MALL_HIT_RATE_1", {}).get("median", float("nan"))
            hr2 = data.get("MALL_HIT_RATE_2", {}).get("median", float("nan"))
            if all1 > 0 or all2 > 0:
                calc1 = hit1 / all1 * 100 if all1 > 0 else float("nan")
                calc2 = hit2 / all2 * 100 if all2 > 0 else float("nan")
                print(f"  {key:15s} {label:5s}  "
                      f"Cache1: hit={fmt(hit1)} all={fmt(all1)} rate={calc1:.2f}% (reported: {fmt(hr1)}%)  "
                      f"Cache2: hit={fmt(hit2)} all={fmt(all2)} rate={calc2:.2f}% (reported: {fmt(hr2)}%)")

    # GEMM kernel counters
    mall_raw = ["MALL_HIT_REQUESTS_1", "MALL_MISS_REQUESTS_1", "MALL_ALL_REQUESTS_1",
                "MALL_HIT_REQUESTS_2", "MALL_MISS_REQUESTS_2", "MALL_ALL_REQUESTS_2"]
    mall_bw = ["MALL_BANDWIDTH_ALL", "MALL_BANDWIDTH_READ_32", "MALL_BANDWIDTH_READ_64",
               "MALL_BANDWIDTH_READ_128", "MALL_BANDWIDTH_WRITE_32", "MALL_BANDWIDTH_WRITE_64"]
    mall_rate = ["MALL_HIT_RATE_1", "MALL_MISS_RATE_1", "MALL_HIT_RATE_2", "MALL_MISS_RATE_2"]
    l2 = ["TCC_HIT_sum", "TCC_MISS_sum"]
    hbm = ["HBM_READ_BYTES", "HBM_WRITE_BYTES"]

    print_section("MALL Raw Requests — GEMM kernel", mall_raw, "GEMM")
    print_section("MALL Hit/Miss Rates — GEMM kernel", mall_rate, "GEMM")
    print_section("MALL Bandwidth — GEMM kernel", mall_bw, "GEMM")
    print_section("L2 Cache — GEMM kernel", l2, "GEMM")
    print_section("HBM — GEMM kernel", hbm, "GEMM")

    print_section("MALL Raw Requests — RCCL kernel", mall_raw, "RCCL")
    print_section("MALL Hit/Miss Rates — RCCL kernel", mall_rate, "RCCL")
    print_section("MALL Bandwidth — RCCL kernel", mall_bw, "RCCL")
    print_section("HBM — RCCL kernel", hbm, "RCCL")

    with open(os.path.join(OUTDIR, "counter_summary_v2.json"), "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"\nJSON: {OUTDIR}/counter_summary_v2.json")


if __name__ == "__main__":
    main()
