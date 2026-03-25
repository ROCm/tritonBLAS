#!/usr/bin/env python3
"""Generate per-problem CU sweep plots comparing WS per-XCD vs WS Global-Atomic."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = "results/plots"
DATA = "results/plot_data"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0D0D0D",
    "axes.facecolor": "#141414",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#DDD",
    "text.color": "#DDD",
    "xtick.color": "#BBB",
    "ytick.color": "#BBB",
    "grid.color": "#333",
    "grid.alpha": 0.5,
    "legend.facecolor": "#1A1A1A",
    "legend.edgecolor": "#555",
    "legend.labelcolor": "#DDD",
    "font.family": "sans-serif",
    "font.size": 11,
})

TORCH_COLOR = "#6BB5FF"
WS_COLOR = "#FF6B8A"
SK_COLOR = "#FFB86B"
GA_COLOR = "#00E5FF"

HIER_COLOR = "#B2FF59"
SERIES = [
    ("torch",           TORCH_COLOR, "s", "torch.matmul (CU-masked)"),
    ("ws_grid",         WS_COLOR,    "o", "Work-Stealing (per-XCD/slot)"),
    ("streamk_static",  SK_COLOR,    "*", "Static Stream-K (grid-limited)"),
    ("ws_global_atomic", GA_COLOR,   "D", "WS Global-Atomic (grid-limited)"),
    ("ws_hierarchical", HIER_COLOR,  "^", "WS Hierarchical (per-XCD + global)"),
]

for M in [2048, 4096, 8192, 12288, 16384]:
    path = f"{DATA}/cu_sweep_{M}_ga.json"
    if not os.path.exists(path):
        print(f"  Skipping {M} (no data)")
        continue

    with open(path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(14, 8))

    for key, color, marker, label in SERIES:
        series = data.get(key, [])
        if not series:
            continue
        pts = sorted(series, key=lambda p: p["cus"])
        cus = [p["cus"] for p in pts]
        tf = [p["tflops"] for p in pts]
        ax.plot(cus, tf, f"{marker}-", color=color, lw=2.5, ms=6, label=label)

    bm = 256 if M >= 4096 else (128 if M >= 2048 else 64)
    tiles = (M // bm) ** 2

    sorted_cus = sorted(set(p["cus"] for s in data.values() for p in s))
    if len(sorted_cus) > 15:
        tick_cus = sorted_cus[::2]
        if sorted_cus[-1] not in tick_cus:
            tick_cus.append(sorted_cus[-1])
    else:
        tick_cus = sorted_cus
    ax.set_xticks(tick_cus)
    ax.set_xticklabels([str(c) for c in tick_cus], fontsize=8, rotation=45)

    ax.set_xlabel("Available CUs", fontsize=13)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=13)
    ax.set_title(f"{M}x{M}x{M} BF16 ({tiles} tiles)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 320)

    fig.tight_layout()
    fname = f"ga_sweep_{M}.png"
    fig.savefig(f"{OUT}/{fname}", dpi=200)
    print(f"Saved {fname}")
    plt.close()

print(f"\nAll plots saved to {OUT}/")
