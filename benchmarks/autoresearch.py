#!/usr/bin/env python3
"""
Autonomous Overlap Research Runner

Inspired by karpathy/autoresearch — runs experiments autonomously,
records results, and generates analysis for the overlap investigation.

Usage:
    # Run a single experiment
    python3 benchmarks/autoresearch.py run --experiment l2-gemm-alone

    # Run the full WS cache investigation
    python3 benchmarks/autoresearch.py suite --suite ws-cache-investigation

    # Parse and analyze collected counter data
    python3 benchmarks/autoresearch.py analyze --results-dir results/ws_cache

    # Show experiment log
    python3 benchmarks/autoresearch.py log
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
EXPERIMENT_LOG = RESULTS_DIR / "experiment_log.json"
WS_CACHE_DIR = RESULTS_DIR / "ws_cache"

# The 4 conditions: {alone, rccl} x {warm, rotating}
CONDITIONS = {
    "alone_warm":     {"profile_mode": "gemm-alone",          "distributed": False},
    "alone_rotating": {"profile_mode": "gemm-rotating",       "distributed": False},
    "rccl_warm":      {"profile_mode": "gemm-rccl",           "distributed": True},
    "rccl_rotating":  {"profile_mode": "gemm-rccl-rotating",  "distributed": True},
}

# Counter groups — each is one rocprofv3 pass
COUNTER_GROUPS = {
    "l2_hitmiss": {
        "counters": ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_READ_sum", "TCC_WRITE_sum"],
        "requires_df": False,
    },
    "dram_traffic": {
        "counters": ["TCC_EA0_RDREQ_DRAM_sum", "TCC_EA0_WRREQ_DRAM_sum",
                      "TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum"],
        "requires_df": False,
    },
    "credit_stalls": {
        "counters": ["TCC_EA0_RDREQ_DRAM_CREDIT_STALL_sum",
                      "TCC_EA0_RDREQ_GMI_CREDIT_STALL_sum"],
        "requires_df": False,
    },
    "mall_bandwidth": {
        "counters": ["MALL_BANDWIDTH_ALL", "HBM_READ_BYTES", "HBM_WRITE_BYTES"],
        "requires_df": True,
    },
    "mall_hitmiss": {
        "counters": ["MALL_ALL_REQUESTS", "MALL_HIT_REQUESTS", "MALL_MISS_REQUESTS"],
        "requires_df": True,
    },
}


def load_log():
    if EXPERIMENT_LOG.exists():
        with open(EXPERIMENT_LOG) as f:
            return json.load(f)
    return {"experiments": []}


def save_log(log):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENT_LOG, "w") as f:
        json.dump(log, f, indent=2)


def record_experiment(experiment_id, hypothesis, config, results, conclusion):
    log = load_log()
    entry = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "hypothesis": hypothesis,
        "configuration": config,
        "results": results,
        "conclusion": conclusion,
    }
    log["experiments"].append(entry)
    save_log(log)
    print(f"  Recorded: {experiment_id}")
    return entry


def run_command(cmd, cwd=None, timeout=600):
    """Run a shell command and return (returncode, stdout, stderr)."""
    short = " ".join(cmd[:8])
    if len(cmd) > 8:
        short += " ..."
    print(f"  CMD: {short}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, cwd=cwd,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"


def parse_rocprof_csv(csv_path):
    """Parse rocprofv3 CSV output and return list of dicts."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def find_counter_csv(out_dir):
    """Find the counter collection CSV in a rocprofv3 output directory."""
    candidates = list(Path(out_dir).rglob("*counter_collection*"))
    if candidates:
        return str(candidates[0])
    candidates = list(Path(out_dir).rglob("*.csv"))
    if candidates:
        return str(candidates[0])
    return None


def sum_counter(rows, counter_name, kernel_filter="ws_persistent_matmul"):
    """Sum a counter across all rows, filtering by kernel name.
    
    rocprofv3 CSV format has Counter_Name/Counter_Value columns (one row per
    kernel+counter pair), not counter names as column headers.
    """
    total = 0.0
    for row in rows:
        if kernel_filter and kernel_filter not in row.get("Kernel_Name", ""):
            continue
        row_counter = row.get("Counter_Name", "")
        if row_counter != counter_name:
            continue
        val = row.get("Counter_Value", 0)
        try:
            total += float(val)
        except (ValueError, TypeError):
            pass
    return total


def run_profiled(label, profile_mode, counters, args, out_base):
    """Run a single profiled experiment: rocprofv3 --pmc <counters> -- overlap.py l2-profile."""
    cond = CONDITIONS[label.split("__")[0]] if "__" in label else CONDITIONS.get(label, {})
    distributed = cond.get("distributed", "rccl" in profile_mode)
    out_dir = out_base / label
    out_dir.mkdir(parents=True, exist_ok=True)

    m = getattr(args, "m", 8192)
    n = getattr(args, "n", 8192)
    k = getattr(args, "k", 8192)
    steps = getattr(args, "steps", 20)
    warmup = getattr(args, "warmup", 10)
    backend = getattr(args, "backend", "ws")
    nproc = getattr(args, "nproc", 8)

    inner_cmd = []
    if distributed:
        inner_cmd = [
            "torchrun", f"--nproc_per_node={nproc}",
            "benchmarks/overlap.py", "l2-profile",
            "--profile-mode", profile_mode,
            "--backend", backend,
            "--gemm-m", str(m), "--gemm-n", str(n), "--gemm-k", str(k),
            "--warmup", str(warmup), "--steps", str(steps),
            "--comm-size", str(m), str(n),
        ]
    else:
        inner_cmd = [
            sys.executable, "benchmarks/overlap.py", "l2-profile",
            "--profile-mode", profile_mode,
            "--backend", backend,
            "--gemm-m", str(m), "--gemm-n", str(n), "--gemm-k", str(k),
            "--warmup", str(warmup), "--steps", str(steps),
        ]

    cmd = [
        "rocprofv3",
        "--pmc", *counters,
        "-o", "out", "-d", str(out_dir),
        "--output-format", "csv",
        "--", *inner_cmd,
    ]

    timeout = 300 if distributed else 180
    rc, stdout, stderr = run_command(cmd, cwd=str(RESULTS_DIR.parent), timeout=timeout)

    timing = ""
    for line in stdout.splitlines():
        if "ms/iter" in line:
            timing = line.strip()
            break

    print(f"    {label}: rc={rc} | {timing}")
    if rc != 0:
        err_tail = stderr[-300:] if stderr else stdout[-300:]
        print(f"    ERROR: {err_tail}")

    return {
        "returncode": rc,
        "timing": timing,
        "csv_dir": str(out_dir),
        "stdout_tail": stdout[-500:],
        "stderr_tail": stderr[-300:] if rc != 0 else "",
    }


# ==============================================================================
# Experiment Definitions
# ==============================================================================

EXPERIMENTS = {}


def experiment(name):
    """Decorator to register an experiment."""
    def decorator(fn):
        EXPERIMENTS[name] = fn
        return fn
    return decorator


def _run_counter_group(group_name, args, hypothesis_prefix=""):
    """Run one counter group across all 4 conditions."""
    group = COUNTER_GROUPS[group_name]
    counters = group["counters"]
    out_base = WS_CACHE_DIR / group_name

    print(f"\n{'='*70}")
    print(f"  Counter group: {group_name}")
    print(f"  Counters: {', '.join(counters)}")
    print(f"{'='*70}")

    results = {}
    for cond_name, cond_cfg in CONDITIONS.items():
        label = f"{cond_name}__{group_name}"
        r = run_profiled(label, cond_cfg["profile_mode"], counters, args, out_base)
        results[cond_name] = r

    parsed = {}
    for cond_name, r in results.items():
        csv_path = find_counter_csv(r["csv_dir"])
        if csv_path:
            rows = parse_rocprof_csv(csv_path)
            sums = {}
            for c in counters:
                sums[c] = sum_counter(rows, c)
            parsed[cond_name] = {"sums": sums, "n_rows": len(rows), "timing": r["timing"]}
        else:
            parsed[cond_name] = {"sums": {}, "n_rows": 0, "timing": r["timing"]}

    _print_comparison(group_name, counters, parsed)

    hypothesis = f"{hypothesis_prefix}Rotating buffers cause measurable change in {group_name} counters"
    conclusion = _auto_conclude(group_name, parsed)

    return record_experiment(
        f"ws-{group_name}",
        hypothesis,
        {
            "group": group_name,
            "counters": counters,
            "conditions": list(CONDITIONS.keys()),
            "m": getattr(args, "m", 8192),
            "n": getattr(args, "n", 8192),
            "k": getattr(args, "k", 8192),
            "backend": getattr(args, "backend", "ws"),
            "steps": getattr(args, "steps", 20),
        },
        {"per_condition": {k: v for k, v in parsed.items()}},
        conclusion,
    )


def _print_comparison(group_name, counters, parsed):
    """Print a formatted comparison table."""
    print(f"\n  {'─'*70}")
    print(f"  RESULTS: {group_name}")
    print(f"  {'─'*70}")

    header = f"  {'Condition':<22}"
    for c in counters:
        short = c.replace("TCC_EA0_", "").replace("_sum", "").replace("_CREDIT_STALL", "_STALL")
        header += f" {short:>16}"
    header += f" {'Timing':>14}"
    print(header)
    print(f"  {'─'*70}")

    for cond_name in CONDITIONS:
        p = parsed.get(cond_name, {})
        sums = p.get("sums", {})
        timing = p.get("timing", "N/A")
        row = f"  {cond_name:<22}"
        for c in counters:
            val = sums.get(c, 0)
            if val > 1e9:
                row += f" {val/1e9:>13.2f}G  "
            elif val > 1e6:
                row += f" {val/1e6:>13.2f}M  "
            elif val > 1e3:
                row += f" {val/1e3:>13.2f}K  "
            else:
                row += f" {val:>16}"
            # trim to 16 chars
        # extract ms from timing
        ms = ""
        if timing:
            for part in timing.split():
                try:
                    float(part)
                    ms = part
                    break
                except ValueError:
                    pass
        row += f" {ms:>10} ms"
        print(row)

    # Derived ratios
    aw = parsed.get("alone_warm", {}).get("sums", {})
    ar = parsed.get("alone_rotating", {}).get("sums", {})
    rw = parsed.get("rccl_warm", {}).get("sums", {})
    rr = parsed.get("rccl_rotating", {}).get("sums", {})

    if aw and ar:
        print(f"\n  Ratios (rotating/warm):")
        for c in counters:
            aw_v = aw.get(c, 0)
            ar_v = ar.get(c, 0)
            rw_v = rw.get(c, 0)
            rr_v = rr.get(c, 0)
            short = c.replace("TCC_EA0_", "").replace("_sum", "").replace("_CREDIT_STALL", "_STALL")
            alone_ratio = ar_v / aw_v if aw_v else float('nan')
            rccl_ratio = rr_v / rw_v if rw_v else float('nan')
            print(f"    {short:<30} alone: {alone_ratio:.3f}x  rccl: {rccl_ratio:.3f}x")

    if "TCC_HIT_sum" in counters:
        print(f"\n  L2 Hit Rates:")
        for cond_name in CONDITIONS:
            s = parsed.get(cond_name, {}).get("sums", {})
            hits = s.get("TCC_HIT_sum", 0)
            misses = s.get("TCC_MISS_sum", 0)
            total = hits + misses
            rate = hits / total * 100 if total else 0
            print(f"    {cond_name:<22} {rate:.2f}% ({hits:,} / {total:,})")


def _auto_conclude(group_name, parsed):
    """Generate an automated conclusion from the parsed data."""
    aw = parsed.get("alone_warm", {}).get("sums", {})
    ar = parsed.get("alone_rotating", {}).get("sums", {})
    rw = parsed.get("rccl_warm", {}).get("sums", {})
    rr = parsed.get("rccl_rotating", {}).get("sums", {})

    if not aw or not ar:
        return "Incomplete data — some conditions failed"

    parts = []
    for c in aw:
        aw_v = aw.get(c, 0)
        ar_v = ar.get(c, 0)
        if aw_v > 0:
            ratio = ar_v / aw_v
            if ratio > 1.1:
                parts.append(f"{c}: rotating {ratio:.2f}x vs warm (alone)")
            elif ratio < 0.9:
                parts.append(f"{c}: rotating {ratio:.2f}x vs warm (alone, LESS)")

    if not parts:
        return f"{group_name}: No significant difference between warm and rotating"
    return f"{group_name}: " + "; ".join(parts)


@experiment("l2-hitmiss")
def exp_l2_hitmiss(args):
    """L2 hit/miss across all 4 conditions."""
    return _run_counter_group("l2_hitmiss", args,
        "WS kernel shows 59% perf drop warm→rotating; L2 miss rate should increase. ")


@experiment("dram-traffic")
def exp_dram_traffic(args):
    """DRAM read/write traffic across all 4 conditions."""
    return _run_counter_group("dram_traffic", args,
        "Rotating buffers force more DRAM reads; RCCL may add GMI traffic. ")


@experiment("credit-stalls")
def exp_credit_stalls(args):
    """DRAM and GMI credit stalls across all 4 conditions."""
    return _run_counter_group("credit_stalls", args,
        "RCCL contention shows as DRAM/GMI credit stalls during overlap. ")


@experiment("mall-bandwidth")
def exp_mall_bandwidth(args):
    """MALL/HBM bandwidth across all 4 conditions (requires df-counters)."""
    return _run_counter_group("mall_bandwidth", args,
        "MALL bandwidth increases with rotating and further with RCCL. ")


@experiment("mall-hitmiss")
def exp_mall_hitmiss(args):
    """MALL hit/miss across all 4 conditions (requires df-counters)."""
    return _run_counter_group("mall_hitmiss", args,
        "MALL miss rate increases with rotating buffers indicating LLC thrashing. ")


# Keep legacy experiments for backward compatibility
@experiment("l2-gemm-alone")
def exp_l2_gemm_alone(args):
    """[Legacy] L2 counters for GEMM alone (warm only)."""
    out_dir = RESULTS_DIR / "l2_gemm_alone"
    out_dir.mkdir(parents=True, exist_ok=True)
    r = run_profiled("alone_warm__legacy", "gemm-alone",
                     ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_READ_sum", "TCC_WRITEBACK_sum"],
                     args, out_dir)
    return record_experiment("l2-gemm-alone", "Baseline L2 for isolated GEMM",
                             {}, {"alone_warm": r}, "Collected")


@experiment("l2-gemm-rccl")
def exp_l2_gemm_rccl(args):
    """[Legacy] L2 counters for GEMM + RCCL (warm only)."""
    out_dir = RESULTS_DIR / "l2_gemm_rccl"
    out_dir.mkdir(parents=True, exist_ok=True)
    r = run_profiled("rccl_warm__legacy", "gemm-rccl",
                     ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_READ_sum", "TCC_WRITEBACK_sum"],
                     args, out_dir)
    return record_experiment("l2-gemm-rccl", "L2 under RCCL overlap",
                             {}, {"rccl_warm": r}, "Collected")


@experiment("size-sweep")
def exp_size_sweep(args):
    """Sweep GEMM size to find where L2/MALL thrashing transitions occur."""
    out_dir = RESULTS_DIR / "size_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = [1024, 2048, 4096, 6144, 8192, 12288, 16384]
    backend = getattr(args, "backend", "ws")

    results = {}
    for size in sizes:
        label = f"alone_{size}"
        cmd = [
            sys.executable, "benchmarks/overlap.py", "l2-profile",
            "--profile-mode", "gemm-alone",
            "--backend", backend,
            "--m", str(size), "--n", str(size), "--k", str(size),
            "--steps", "20",
        ]
        rc, stdout, stderr = run_command(cmd, cwd=str(RESULTS_DIR.parent))
        results[label] = {"returncode": rc, "stdout": stdout[-300:]}
        print(f"    {label}: rc={rc}")

    return record_experiment(
        "size-sweep",
        "Identify GEMM size thresholds where L2 -> MALL -> HBM transitions occur",
        {"backend": backend, "sizes": sizes},
        results, "Size sweep completed",
    )


# ==============================================================================
# Experiment Suites
# ==============================================================================

SUITES = {
    "ws-cache-investigation": [
        "l2-hitmiss",
        "dram-traffic",
        "credit-stalls",
    ],
    "ws-cache-full": [
        "l2-hitmiss",
        "dram-traffic",
        "credit-stalls",
        "mall-bandwidth",
        "mall-hitmiss",
    ],
    "l2-thrashing": [
        "l2-gemm-alone",
        "l2-gemm-rccl",
    ],
    "mall-thrashing": [
        "mall-bandwidth",
        "mall-hitmiss",
    ],
    "full-cache-analysis": [
        "l2-hitmiss",
        "dram-traffic",
        "credit-stalls",
        "mall-bandwidth",
        "mall-hitmiss",
        "size-sweep",
    ],
}


# ==============================================================================
# CLI
# ==============================================================================

def cmd_run(args):
    """Run a single experiment."""
    name = args.experiment
    if name not in EXPERIMENTS:
        print(f"Unknown experiment: {name}")
        print(f"Available: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)
    print(f"\n{'='*70}")
    print(f"Running experiment: {name}")
    print(f"{'='*70}")
    EXPERIMENTS[name](args)


def cmd_suite(args):
    """Run an experiment suite."""
    name = args.suite
    if name not in SUITES:
        print(f"Unknown suite: {name}")
        print(f"Available: {', '.join(SUITES.keys())}")
        sys.exit(1)
    experiments = SUITES[name]
    print(f"\n{'='*70}")
    print(f"AUTORESEARCH SUITE: {name} ({len(experiments)} experiments)")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"{'='*70}")
    for i, exp_name in enumerate(experiments, 1):
        print(f"\n{'─'*70}")
        print(f"  [{i}/{len(experiments)}] {exp_name}")
        print(f"{'─'*70}")
        EXPERIMENTS[exp_name](args)
    print(f"\n{'='*70}")
    print(f"SUITE COMPLETE: {name}")
    print(f"Results in: {WS_CACHE_DIR}")
    print(f"{'='*70}")


def cmd_log(args):
    """Show experiment log."""
    log = load_log()
    if not log["experiments"]:
        print("No experiments recorded yet.")
        return
    for exp in log["experiments"]:
        print(f"\n{'='*60}")
        print(f"  ID:         {exp['experiment_id']}")
        print(f"  Time:       {exp['timestamp']}")
        print(f"  Hypothesis: {exp['hypothesis']}")
        print(f"  Conclusion: {exp['conclusion']}")


def cmd_analyze(args):
    """Analyze collected counter data from ws_cache directory."""
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print(f"Analyzing: {results_dir}\n")

    for group_dir in sorted(results_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        group_name = group_dir.name
        group_cfg = COUNTER_GROUPS.get(group_name)
        if not group_cfg:
            continue

        counters = group_cfg["counters"]
        parsed = {}

        for cond_dir in sorted(group_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            cond_name = cond_dir.name.split("__")[0]
            csv_path = find_counter_csv(str(cond_dir))
            if csv_path:
                rows = parse_rocprof_csv(csv_path)
                sums = {c: sum_counter(rows, c) for c in counters}
                parsed[cond_name] = {"sums": sums, "n_rows": len(rows), "timing": ""}

        if parsed:
            _print_comparison(group_name, counters, parsed)


def cmd_list(args):
    """List available experiments and suites."""
    print("Available experiments:")
    for name in EXPERIMENTS:
        fn = EXPERIMENTS[name]
        print(f"  {name:25s} -- {fn.__doc__ or ''}")
    print("\nAvailable suites:")
    for name, exps in SUITES.items():
        print(f"  {name:25s} -- {', '.join(exps)}")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Overlap Research Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a single experiment")
    p_run.add_argument("--experiment", "-e", required=True, help="Experiment name")
    p_run.add_argument("--backend", default="ws")
    p_run.add_argument("--steps", type=int, default=20)
    p_run.add_argument("--warmup", type=int, default=10)
    p_run.add_argument("--nproc", type=int, default=8)
    p_run.add_argument("--m", type=int, default=8192)
    p_run.add_argument("--n", type=int, default=8192)
    p_run.add_argument("--k", type=int, default=8192)

    p_suite = sub.add_parser("suite", help="Run an experiment suite")
    p_suite.add_argument("--suite", "-s", required=True, help="Suite name")
    p_suite.add_argument("--backend", default="ws")
    p_suite.add_argument("--steps", type=int, default=20)
    p_suite.add_argument("--warmup", type=int, default=10)
    p_suite.add_argument("--nproc", type=int, default=8)
    p_suite.add_argument("--m", type=int, default=8192)
    p_suite.add_argument("--n", type=int, default=8192)
    p_suite.add_argument("--k", type=int, default=8192)

    p_log = sub.add_parser("log", help="Show experiment log")

    p_analyze = sub.add_parser("analyze", help="Analyze collected data")
    p_analyze.add_argument("--results-dir", default=str(WS_CACHE_DIR))

    p_list = sub.add_parser("list", help="List experiments and suites")

    args = parser.parse_args()
    dispatch = {
        "run": cmd_run,
        "suite": cmd_suite,
        "log": cmd_log,
        "analyze": cmd_analyze,
        "list": cmd_list,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
