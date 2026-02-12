#!/usr/bin/env python3
"""
Plot CU-sweep results: one line per kernel variant.

Usage:
    python tools/plot_cu_sweep.py \
        --persistent results/cu_sweep_persistent_full.csv \
        --streamk    results/cu_sweep_streamk_full.csv \
        --torch      results/cu_sweep_torch_full.csv \
        --ws-cpc 1  results/cu_sweep_ws_cpc1.csv \
        --ws-cpc 2  results/cu_sweep_ws_cpc2.csv \
        --ws-cpc 4  results/cu_sweep_ws_cpc4.csv \
        --ws-cpc 8  results/cu_sweep_ws_cpc8.csv \
        --ws-cpc 16 results/cu_sweep_ws_cpc16.csv \
        -o results/cu_sweep_plot.png
"""
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_tritonblas_csv(path):
    """Load a tritonblas benchmark CSV, return sorted (active_cus, gflops) lists."""
    cus, gflops = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            cus.append(int(row["active_cus"]))
            gflops.append(float(row["tritonblas_gflops"]))
    order = sorted(range(len(cus)), key=lambda i: cus[i])
    return [cus[i] for i in order], [gflops[i] for i in order]


def load_torch_csv(path):
    """Load a torch.mm benchmark CSV, return sorted (active_cus, gflops) lists."""
    cus, gflops = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            cus.append(int(row["active_cus"]))
            gflops.append(float(row["gflops"]))
    order = sorted(range(len(cus)), key=lambda i: cus[i])
    return [cus[i] for i in order], [gflops[i] for i in order]


def main():
    parser = argparse.ArgumentParser(description="Plot CU-sweep benchmark results.")
    parser.add_argument("--persistent", type=str, help="Persistent GEMM CSV")
    parser.add_argument("--streamk", type=str, help="Stream-K GEMM CSV")
    parser.add_argument("--ws", type=str, help="Work-stealing GEMM CSV (single line, legacy)")
    parser.add_argument("--ws-cpc", nargs=2, action="append", metavar=("CPC", "CSV"),
                        help="Work-stealing CSV with counters-per-XCD value, e.g. --ws-cpc 4 file.csv")
    parser.add_argument("--torch", type=str, help="torch.mm CSV")
    parser.add_argument("-o", "--output", type=str, default="cu_sweep_plot.png",
                        help="Output image path (default: cu_sweep_plot.png)")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom plot title (default: auto-generated)")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(14, 8))

    if args.persistent:
        cus, gf = load_tritonblas_csv(args.persistent)
        ax.plot(cus, gf, label="Persistent", linewidth=2.5, markersize=5,
                color="#2196F3", marker="o")
    if args.streamk:
        cus, gf = load_tritonblas_csv(args.streamk)
        ax.plot(cus, gf, label="Stream-K", linewidth=2.5, markersize=5,
                color="#4CAF50", marker="s")
    if args.torch:
        cus, gf = load_torch_csv(args.torch)
        ax.plot(cus, gf, label="torch.mm (hipBLASLt)", linewidth=2.5, markersize=5,
                color="#F44336", marker="D")

    ws_colors = {
        "1":  "#FF9800",
        "2":  "#9C27B0",
        "4":  "#00BCD4",
        "8":  "#795548",
        "16": "#E91E63",
    }
    ws_markers = {
        "1": "^", "2": "v", "4": "<", "8": ">", "16": "P",
    }

    if args.ws and not args.ws_cpc:
        cus, gf = load_tritonblas_csv(args.ws)
        ax.plot(cus, gf, label="Work-Stealing", linewidth=2, markersize=5,
                color="#FF9800", marker="^")

    if args.ws_cpc:
        for cpc_val, csv_path in sorted(args.ws_cpc, key=lambda x: int(x[0])):
            cus, gf = load_tritonblas_csv(csv_path)
            color = ws_colors.get(cpc_val, "#607D8B")
            marker = ws_markers.get(cpc_val, "x")
            ax.plot(cus, gf, label=f"Work-Stealing (Counters/XCD = {cpc_val})", linewidth=1.8, markersize=5,
                    color=color, marker=marker, linestyle="--")

    ax.set_xlabel("Active CUs", fontsize=13)
    ax.set_ylabel("GFLOPS", fontsize=13)
    title = args.title if args.title else "FP16 GEMM — CU Sweep (MI300X)"
    ax.set_title(title, fontsize=15)
    ax.legend(fontsize=11, loc="upper left", ncol=2)
    ax.set_xticks(np.arange(32, 312, 8))
    ax.set_xlim(32, 312)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
