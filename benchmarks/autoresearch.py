#!/usr/bin/env python3
"""
Autonomous Overlap Research Runner

Inspired by karpathy/autoresearch — runs experiments autonomously,
records results, and generates analysis for the overlap investigation.

This is the experiment runner that the AI agent (or human) drives.
It manages the experiment lifecycle: run, collect, analyze, record.

Usage:
    # Run a single experiment
    python3 benchmarks/autoresearch.py run --experiment l2-baseline

    # Run the full experiment suite
    python3 benchmarks/autoresearch.py suite --suite l2-thrashing

    # Parse and analyze collected counter data
    python3 benchmarks/autoresearch.py analyze --results-dir results/

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
    print(f"Recorded experiment: {experiment_id}")
    return entry


def run_command(cmd, cwd=None, timeout=600):
    """Run a shell command and return (returncode, stdout, stderr)."""
    print(f"  Running: {' '.join(cmd[:6])}...")
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


def compute_l2_hit_rate(rows):
    """Compute L2 hit rate from TCC_HIT and TCC_MISS counters."""
    total_hit = 0
    total_miss = 0
    for row in rows:
        total_hit += int(row.get("TCC_HIT", 0))
        total_miss += int(row.get("TCC_MISS", 0))
    total = total_hit + total_miss
    if total == 0:
        return 0.0
    return total_hit / total * 100


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


@experiment("l2-gemm-alone")
def exp_l2_gemm_alone(args):
    """Measure L2 counters for GEMM running in isolation."""
    out_dir = RESULTS_DIR / "l2_gemm_alone"
    out_dir.mkdir(parents=True, exist_ok=True)

    backends = getattr(args, "backends", ["ws"])
    sizes = getattr(args, "sizes", [(8192, 8192, 8192)])

    results = {}
    for backend in backends:
        for m, n, k in sizes:
            label = f"{backend}_{m}x{n}x{k}"
            cmd = [
                "rocprofv3",
                "--pmc", "TCC_HIT", "TCC_MISS", "TCC_READ", "TCC_WRITEBACK",
                "-o", "out", "-d", str(out_dir / label),
                "--output-format", "csv",
                "--",
                sys.executable, "benchmarks/overlap.py", "l2-profile",
                "--profile-mode", "gemm-alone",
                "--backend", backend,
                "--m", str(m), "--n", str(n), "--k", str(k),
                "--steps", str(getattr(args, "steps", 20)),
            ]
            rc, stdout, stderr = run_command(cmd, cwd=str(RESULTS_DIR.parent))
            results[label] = {
                "returncode": rc,
                "stdout": stdout[-500:],
                "stderr": stderr[-500:] if rc != 0 else "",
            }
            print(f"    {label}: rc={rc}")

    return record_experiment(
        "l2-gemm-alone",
        "Establish baseline L2 hit rates for GEMM running in isolation",
        {"backends": backends, "sizes": [list(s) for s in sizes]},
        results,
        "Baseline L2 counters collected",
    )


@experiment("l2-gemm-rccl")
def exp_l2_gemm_rccl(args):
    """Measure L2 counters for GEMM + RCCL overlap."""
    out_dir = RESULTS_DIR / "l2_gemm_rccl"
    out_dir.mkdir(parents=True, exist_ok=True)

    backends = getattr(args, "backends", ["ws"])
    sizes = getattr(args, "sizes", [(8192, 8192, 8192)])
    nproc = getattr(args, "nproc", 8)

    results = {}
    for backend in backends:
        for m, n, k in sizes:
            label = f"{backend}_{m}x{n}x{k}"
            cmd = [
                "rocprofv3",
                "--pmc", "TCC_HIT", "TCC_MISS", "TCC_READ", "TCC_WRITEBACK",
                "-o", "out", "-d", str(out_dir / label),
                "--output-format", "csv",
                "--",
                "torchrun", f"--nproc_per_node={nproc}",
                "benchmarks/overlap.py", "l2-profile",
                "--profile-mode", "gemm-rccl",
                "--backend", backend,
                "--m", str(m), "--n", str(n), "--k", str(k),
                "--steps", str(getattr(args, "steps", 20)),
            ]
            rc, stdout, stderr = run_command(cmd, cwd=str(RESULTS_DIR.parent), timeout=300)
            results[label] = {
                "returncode": rc,
                "stdout": stdout[-500:],
                "stderr": stderr[-500:] if rc != 0 else "",
            }
            print(f"    {label}: rc={rc}")

    return record_experiment(
        "l2-gemm-rccl",
        "L2 hit rate drops during GEMM+RCCL overlap due to RCCL cache evictions",
        {"backends": backends, "sizes": [list(s) for s in sizes], "nproc": nproc},
        results,
        "L2 counters under overlap collected",
    )


@experiment("mall-gemm-alone")
def exp_mall_gemm_alone(args):
    """Measure MALL/LLC + HBM counters for GEMM in isolation (requires df-counters)."""
    out_dir = RESULTS_DIR / "mall_gemm_alone"
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = getattr(args, "sizes", [(8192, 8192, 8192)])
    backend = getattr(args, "backend", "ws")

    results = {}
    for m, n, k in sizes:
        label = f"{backend}_{m}x{n}x{k}"
        cmd = [
            "rocprofv3",
            "--pmc", "MALL_BANDWIDTH_ALL", "HBM_READ_BYTES", "HBM_WRITE_BYTES",
            "-o", "out", "-d", str(out_dir / label),
            "--output-format", "csv",
            "--",
            sys.executable, "benchmarks/overlap.py", "l2-profile",
            "--profile-mode", "gemm-alone",
            "--backend", backend,
            "--m", str(m), "--n", str(n), "--k", str(k),
            "--steps", str(getattr(args, "steps", 20)),
        ]
        rc, stdout, stderr = run_command(cmd, cwd=str(RESULTS_DIR.parent))
        results[label] = {
            "returncode": rc,
            "stdout": stdout[-500:],
            "stderr": stderr[-500:] if rc != 0 else "",
        }
        print(f"    {label}: rc={rc}")

    return record_experiment(
        "mall-gemm-alone",
        "Establish baseline MALL/LLC and HBM bandwidth for isolated GEMM",
        {"backend": backend, "sizes": [list(s) for s in sizes]},
        results,
        "MALL/HBM baseline counters collected",
    )


@experiment("mall-gemm-rccl")
def exp_mall_gemm_rccl(args):
    """Measure MALL/LLC + HBM counters for GEMM + RCCL (requires df-counters)."""
    out_dir = RESULTS_DIR / "mall_gemm_rccl"
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = getattr(args, "sizes", [(8192, 8192, 8192)])
    backend = getattr(args, "backend", "ws")
    nproc = getattr(args, "nproc", 8)

    results = {}
    for m, n, k in sizes:
        label = f"{backend}_{m}x{n}x{k}"
        cmd = [
            "rocprofv3",
            "--pmc", "MALL_BANDWIDTH_ALL", "HBM_READ_BYTES", "HBM_WRITE_BYTES",
            "-o", "out", "-d", str(out_dir / label),
            "--output-format", "csv",
            "--",
            "torchrun", f"--nproc_per_node={nproc}",
            "benchmarks/overlap.py", "l2-profile",
            "--profile-mode", "gemm-rccl",
            "--backend", backend,
            "--m", str(m), "--n", str(n), "--k", str(k),
            "--steps", str(getattr(args, "steps", 20)),
        ]
        rc, stdout, stderr = run_command(cmd, cwd=str(RESULTS_DIR.parent), timeout=300)
        results[label] = {
            "returncode": rc,
            "stdout": stdout[-500:],
            "stderr": stderr[-500:] if rc != 0 else "",
        }
        print(f"    {label}: rc={rc}")

    return record_experiment(
        "mall-gemm-rccl",
        "RCCL overlap causes LLC/MALL thrashing beyond L2, increasing HBM traffic",
        {"backend": backend, "sizes": [list(s) for s in sizes], "nproc": nproc},
        results,
        "MALL/HBM overlap counters collected",
    )


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
        "Identify GEMM size thresholds where L2 → MALL → HBM transitions occur",
        {"backend": backend, "sizes": sizes},
        results,
        "Size sweep completed",
    )


# ==============================================================================
# Experiment Suites
# ==============================================================================

SUITES = {
    "l2-thrashing": [
        "l2-gemm-alone",
        "l2-gemm-rccl",
    ],
    "mall-thrashing": [
        "mall-gemm-alone",
        "mall-gemm-rccl",
    ],
    "full-cache-analysis": [
        "l2-gemm-alone",
        "l2-gemm-rccl",
        "mall-gemm-alone",
        "mall-gemm-rccl",
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
    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"{'='*60}")
    EXPERIMENTS[name](args)


def cmd_suite(args):
    """Run an experiment suite."""
    name = args.suite
    if name not in SUITES:
        print(f"Unknown suite: {name}")
        print(f"Available: {', '.join(SUITES.keys())}")
        sys.exit(1)
    experiments = SUITES[name]
    print(f"\n{'='*60}")
    print(f"Running suite: {name} ({len(experiments)} experiments)")
    print(f"{'='*60}")
    for exp_name in experiments:
        print(f"\n--- Experiment: {exp_name} ---")
        EXPERIMENTS[exp_name](args)


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
    """Analyze collected counter data."""
    results_dir = Path(args.results_dir)
    print(f"Analyzing results in: {results_dir}")

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        csv_files = list(subdir.rglob("*.csv"))
        if csv_files:
            print(f"\n--- {subdir.name} ({len(csv_files)} CSV files) ---")
            for csv_file in csv_files[:3]:
                rows = parse_rocprof_csv(str(csv_file))
                if rows:
                    print(f"  {csv_file.name}: {len(rows)} rows")
                    cols = [c for c in rows[0].keys() if c.startswith("TCC") or c.startswith("MALL") or c.startswith("HBM")]
                    if cols:
                        print(f"  Counters: {', '.join(cols)}")


def cmd_list(args):
    """List available experiments and suites."""
    print("Available experiments:")
    for name in EXPERIMENTS:
        fn = EXPERIMENTS[name]
        print(f"  {name:25s} — {fn.__doc__ or ''}")
    print("\nAvailable suites:")
    for name, exps in SUITES.items():
        print(f"  {name:25s} — {', '.join(exps)}")


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
    p_run.add_argument("--nproc", type=int, default=8)

    p_suite = sub.add_parser("suite", help="Run an experiment suite")
    p_suite.add_argument("--suite", "-s", required=True, help="Suite name")
    p_suite.add_argument("--backend", default="ws")
    p_suite.add_argument("--steps", type=int, default=20)
    p_suite.add_argument("--nproc", type=int, default=8)

    p_log = sub.add_parser("log", help="Show experiment log")

    p_analyze = sub.add_parser("analyze", help="Analyze collected data")
    p_analyze.add_argument("--results-dir", default=str(RESULTS_DIR))

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
