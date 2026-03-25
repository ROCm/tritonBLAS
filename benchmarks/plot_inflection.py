#!/usr/bin/env python3
"""Generate inflection analysis plots from overlap experiment data."""
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    from matplotlib import font_manager
    for name in ["Helvetica", "Arial", "Liberation Sans"]:
        matches = font_manager.findSystemFonts(fontpaths=None)
        if any(name.lower() in f.lower() for f in matches):
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name]
            break
except Exception:
    pass

STYLE = {
    "figure.facecolor": "#0D1117",
    "axes.facecolor": "#0D1117",
    "text.color": "#C9D1D9",
    "axes.labelcolor": "#C9D1D9",
    "xtick.color": "#8B949E",
    "ytick.color": "#8B949E",
    "axes.edgecolor": "#30363D",
    "grid.color": "#21262D",
    "grid.alpha": 0.6,
    "legend.facecolor": "#161B22",
    "legend.edgecolor": "#30363D",
}
plt.rcParams.update(STYLE)

TORCH_COLOR = "#FF6B6B"
WS_COLOR = "#4ECDC4"
COMM_COLOR = "#F7DC6F"
IDEAL_COLOR = "#888888"

os.makedirs("results/plots", exist_ok=True)


def load_data(path):
    with open(path) as f:
        return json.load(f)


def plot_penalty_comparison():
    """Bar chart: overlap penalty across sizes."""
    files = {
        "4K": "results/overlap_data/inflection_overlap_4k.json",
        "8K": "results/overlap_data/inflection_overlap_8k.json",
        "12K": "results/overlap_data/inflection_overlap.json",
        "16K": "results/overlap_data/inflection_overlap_16k.json",
    }

    labels, torch_penalties, ws_penalties = [], [], []
    for label, path in files.items():
        try:
            d = load_data(path)
            labels.append(label)
            torch_penalties.append(d["torch"]["penalty_pct"])
            best_ws = min(d["ws_hierarchical"].values(), key=lambda x: x["overlap_wall_ms"])
            ws_penalties.append(best_ws["penalty_pct"])
        except Exception as e:
            print(f"Skip {label}: {e}")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    w = 0.35

    bars_t = ax.bar(x - w/2, torch_penalties, w, label="torch.matmul", color=TORCH_COLOR, edgecolor="none")
    bars_ws = ax.bar(x + w/2, ws_penalties, w, label="WS Hierarchical (best CU)", color=WS_COLOR, edgecolor="none")

    for bar, val in zip(bars_t, torch_penalties):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, color=TORCH_COLOR, fontweight="bold")
    for bar, val in zip(bars_ws, ws_penalties):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, color=WS_COLOR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("RCCL Overlap Penalty (%)", fontsize=13)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(torch_penalties) * 1.25)

    fig.tight_layout()
    fig.savefig("results/plots/inflection_penalty.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: results/plots/inflection_penalty.png")


def plot_wall_comparison():
    """Bar chart: actual overlap wall time (who wins?)."""
    files = {
        "4K": "results/overlap_data/inflection_overlap_4k.json",
        "8K": "results/overlap_data/inflection_overlap_8k.json",
        "12K": "results/overlap_data/inflection_overlap.json",
        "16K": "results/overlap_data/inflection_overlap_16k.json",
    }

    labels = []
    torch_walls, ws_walls, ideal_walls = [], [], []

    for label, path in files.items():
        try:
            d = load_data(path)
            labels.append(label)
            torch_walls.append(d["torch"]["overlap_wall_ms"])
            best_ws = min(d["ws_hierarchical"].values(), key=lambda x: x["overlap_wall_ms"])
            ws_walls.append(best_ws["overlap_wall_ms"])
            ideal_walls.append(max(d["torch"]["alone_ms"], d["comm_alone_ms"]))
        except Exception as e:
            print(f"Skip {label}: {e}")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    w = 0.25

    ax.bar(x - w, ideal_walls, w, label="Ideal (max of GEMM, comm)", color=IDEAL_COLOR, alpha=0.5, edgecolor="none")
    bars_t = ax.bar(x, torch_walls, w, label="torch.matmul", color=TORCH_COLOR, edgecolor="none")
    bars_ws = ax.bar(x + w, ws_walls, w, label="WS Hierarchical (best CU)", color=WS_COLOR, edgecolor="none")

    for i, (tw, wsw) in enumerate(zip(torch_walls, ws_walls)):
        diff_pct = (wsw - tw) / tw * 100
        color = WS_COLOR if wsw < tw else TORCH_COLOR
        marker = "WS wins" if wsw < tw else "torch wins"
        ypos = max(tw, wsw) + 0.05 * max(torch_walls)
        ax.text(x[i], ypos, f"{marker}\n({diff_pct:+.1f}%)", ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Overlap Wall Time (ms)", fontsize=13)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/plots/inflection_wall.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: results/plots/inflection_wall.png")


def plot_cu_sweep_with_overlap_band():
    """For 8K: show CU sweep curves with the RCCL effective-CU region highlighted."""
    with open("results/plot_data/cu_sweep_8192_ga.json") as f:
        sweep = json.load(f)
    with open("results/plot_data/hierarchical_100pct_all.json") as f:
        hier_all = json.load(f)

    torch_pts = sorted(sweep["torch"], key=lambda d: d["cus"])
    hier_pts = sorted(hier_all["8192"], key=lambda d: d["cus"])

    # Overlap data for 8K small comm
    with open("results/overlap_data/inflection_overlap_8k.json") as f:
        olap = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 7))

    tcus = [d["cus"] for d in torch_pts]
    ttf = [d["tflops"] for d in torch_pts]
    hcus = [d["cus"] for d in hier_pts]
    htf = [d["tflops"] for d in hier_pts]

    ax.plot(tcus, ttf, "o-", color=TORCH_COLOR, label="torch.matmul (CU masked)", linewidth=2, markersize=4)
    ax.plot(hcus, htf, "s-", color=WS_COLOR, label="WS Hierarchical (grid-limited)", linewidth=2, markersize=4)

    # Mark the overlap operating points
    torch_overlap_tf = olap["torch"]["overlap_gemm_tf"]  # effective TF during overlap
    ax.axhline(y=torch_overlap_tf, color=TORCH_COLOR, linestyle="--", alpha=0.5, linewidth=1)
    ax.annotate(f"torch overlap\n{torch_overlap_tf:.0f} TF",
                xy=(50, torch_overlap_tf), fontsize=10, color=TORCH_COLOR,
                ha="left", va="bottom")

    best_ws_key = min(olap["ws_hierarchical"], key=lambda k: olap["ws_hierarchical"][k]["overlap_wall_ms"])
    ws_overlap_tf = olap["ws_hierarchical"][best_ws_key]["overlap_gemm_tf"]
    ax.axhline(y=ws_overlap_tf, color=WS_COLOR, linestyle="--", alpha=0.5, linewidth=1)
    ax.annotate(f"WS overlap @{best_ws_key} CUs\n{ws_overlap_tf:.0f} TF",
                xy=(50, ws_overlap_tf), fontsize=10, color=WS_COLOR,
                ha="left", va="bottom")

    # Shade the "RCCL contention zone" where effective CUs drop
    ax.axvspan(256, 304, alpha=0.08, color="#F7DC6F", label="RCCL contention zone (approx)")

    ax.set_xlabel("Active Compute Units (CUs)", fontsize=13)
    ax.set_ylabel("Throughput (TFLOPs)", fontsize=13)
    ax.set_xlim(0, 312)
    ax.set_xticks(range(0, 312, 16))
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/plots/inflection_8k_cusweep.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: results/plots/inflection_8k_cusweep.png")


def plot_inflection_detail():
    """Detailed 8K overlap: alone vs overlap for both backends at various CU counts."""
    with open("results/overlap_data/inflection_overlap_8k.json") as f:
        d = load_data("results/overlap_data/inflection_overlap_8k.json")

    flops = 2.0 * 8192**3

    # Collect WS data
    cus = sorted([int(k) for k in d["ws_hierarchical"]])
    ws_alone = [d["ws_hierarchical"][str(c)]["alone_ms"] for c in cus]
    ws_overlap = [d["ws_hierarchical"][str(c)]["overlap_gemm_ms"] for c in cus]
    ws_wall = [d["ws_hierarchical"][str(c)]["overlap_wall_ms"] for c in cus]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: GEMM latency (alone vs overlap)
    ax1.plot(cus, ws_alone, "s-", color=WS_COLOR, label="WS alone", linewidth=2, markersize=6)
    ax1.plot(cus, ws_overlap, "s--", color=WS_COLOR, label="WS + RCCL (GEMM)", linewidth=2, markersize=6, alpha=0.7)
    ax1.axhline(y=d["torch"]["alone_ms"], color=TORCH_COLOR, linestyle="-", linewidth=2, label=f'torch alone ({d["torch"]["alone_ms"]:.3f} ms)')
    ax1.axhline(y=d["torch"]["overlap_gemm_ms"], color=TORCH_COLOR, linestyle="--", linewidth=2, alpha=0.7, label=f'torch + RCCL ({d["torch"]["overlap_gemm_ms"]:.3f} ms)')

    ax1.set_xlabel("Active CUs for WS", fontsize=13)
    ax1.set_ylabel("GEMM Latency (ms)", fontsize=13)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(cus)

    # Right: Overlap wall time
    ax2.plot(cus, ws_wall, "s-", color=WS_COLOR, label="WS wall", linewidth=2, markersize=6)
    ax2.axhline(y=d["torch"]["overlap_wall_ms"], color=TORCH_COLOR, linestyle="-", linewidth=2,
                label=f'torch wall ({d["torch"]["overlap_wall_ms"]:.3f} ms)')
    ax2.axhline(y=d["comm_alone_ms"], color=COMM_COLOR, linestyle=":", linewidth=2, alpha=0.7,
                label=f'comm alone ({d["comm_alone_ms"]:.3f} ms)')

    for i, c in enumerate(cus):
        if ws_wall[i] < d["torch"]["overlap_wall_ms"]:
            ax2.annotate(f"WS wins\n{d['torch']['overlap_wall_ms'] - ws_wall[i]:.3f} ms",
                        xy=(c, ws_wall[i]), xytext=(c, ws_wall[i] - 0.05),
                        fontsize=9, color=WS_COLOR, ha="center", va="top", fontweight="bold")

    ax2.set_xlabel("Active CUs for WS", fontsize=13)
    ax2.set_ylabel("Overlap Wall Time (ms)", fontsize=13)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(cus)

    fig.suptitle("8192x8192x8192 BF16 + all_reduce(8Kx8K), NCH=32", fontsize=14, color="#C9D1D9")
    fig.tight_layout()
    fig.savefig("results/plots/inflection_8k_detail.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: results/plots/inflection_8k_detail.png")


def plot_summary_table():
    """Create a comprehensive summary plot."""
    files = [
        ("4K", "results/overlap_data/inflection_overlap_4k.json"),
        ("8K", "results/overlap_data/inflection_overlap_8k.json"),
        ("12K", "results/overlap_data/inflection_overlap.json"),
        ("16K", "results/overlap_data/inflection_overlap_16k.json"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Inflection Point Analysis: WS Hierarchical vs torch.matmul under RCCL Overlap",
                 fontsize=15, color="#C9D1D9", y=0.98)

    for idx, (label, path) in enumerate(files):
        ax = axes[idx // 2][idx % 2]
        try:
            d = load_data(path)
        except Exception:
            ax.set_visible(False)
            continue

        cus = sorted([int(k) for k in d["ws_hierarchical"]])
        ws_walls = [d["ws_hierarchical"][str(c)]["overlap_wall_ms"] for c in cus]
        ws_alone = [d["ws_hierarchical"][str(c)]["alone_ms"] for c in cus]
        torch_wall = d["torch"]["overlap_wall_ms"]
        torch_alone = d["torch"]["alone_ms"]

        ax.plot(cus, ws_walls, "s-", color=WS_COLOR, linewidth=2, markersize=5, label="WS overlap wall")
        ax.plot(cus, ws_alone, "s:", color=WS_COLOR, linewidth=1.5, markersize=4, alpha=0.5, label="WS alone")
        ax.axhline(y=torch_wall, color=TORCH_COLOR, linewidth=2, label=f"torch overlap wall ({torch_wall:.3f} ms)")
        ax.axhline(y=torch_alone, color=TORCH_COLOR, linewidth=1.5, linestyle=":", alpha=0.5,
                   label=f"torch alone ({torch_alone:.3f} ms)")
        ax.axhline(y=d["comm_alone_ms"], color=COMM_COLOR, linewidth=1, linestyle="--", alpha=0.5,
                   label=f"comm alone ({d['comm_alone_ms']:.3f} ms)")

        wins = [c for c, w in zip(cus, ws_walls) if w < torch_wall]
        if wins:
            best_cu = min(cus, key=lambda c: d["ws_hierarchical"][str(c)]["overlap_wall_ms"])
            best_wall = d["ws_hierarchical"][str(best_cu)]["overlap_wall_ms"]
            savings = (torch_wall - best_wall) / torch_wall * 100
            ax.set_title(f"{label} — WS WINS at CUs {wins} (best: {savings:.1f}% faster)",
                        fontsize=11, color=WS_COLOR, fontweight="bold")
        else:
            ax.set_title(f"{label} — torch wins (WS raw gap too large)", fontsize=11, color=TORCH_COLOR)

        ax.set_xlabel("Active CUs", fontsize=11)
        ax.set_ylabel("Time (ms)", fontsize=11)
        ax.set_xticks(cus)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("results/plots/inflection_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: results/plots/inflection_summary.png")


if __name__ == "__main__":
    plot_penalty_comparison()
    plot_wall_comparison()
    plot_cu_sweep_with_overlap_band()
    plot_inflection_detail()
    plot_summary_table()
