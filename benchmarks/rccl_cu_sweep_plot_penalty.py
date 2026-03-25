#!/usr/bin/env python3
"""Plot RCCL CU sweep: GEMM overlap penalty (%) as RCCL claims more CUs."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_FILE = "results/rccl_sweep/raw_8192.jsonl"
OUT_FILE = "results/plots/rccl_cu_sweep_8k_penalty.png"

with open(DATA_FILE) as f:
    data = [json.loads(line) for line in f if line.strip()]

data = [d for d in data if d["channels"] <= 64]
data.sort(key=lambda d: d["channels"])

channels = np.array([d["channels"] for d in data])
torch_alone = np.array([d["torch_alone_ms"] for d in data])
ws_alone = np.array([d["ws_alone_ms"] for d in data])
torch_gemm_ov = np.array([d["torch_gemm_overlap_ms"] for d in data])
ws_gemm_ov = np.array([d["ws_gemm_overlap_ms"] for d in data])

torch_penalty = (torch_gemm_ov - torch_alone) / torch_alone * 100
ws_penalty = (ws_gemm_ov - ws_alone) / ws_alone * 100

TORCH_COLOR = "#4ADE80"
WS_COLOR = "#FF6B8A"

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

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(channels, torch_penalty, "s-", color=TORCH_COLOR, linewidth=2.5,
        markersize=7, label="torch.matmul", zorder=5)
ax.plot(channels, ws_penalty, "o-", color=WS_COLOR, linewidth=2.5,
        markersize=7, label="Work-Stealing (Hierarchical)", zorder=5)

ax.set_xticks(channels)
ax.set_xticklabels([str(c) for c in channels], fontsize=10)

ax.set_xlabel("NCCL_MAX_NCHANNELS", fontsize=13)
ax.set_ylabel("GEMM Performance Loss (%)", fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc="upper left")

fig.tight_layout()
fig.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
print(f"Saved {OUT_FILE}")

print(f"\n{'Ch':>4} {'T.pen%':>8} {'WS.pen%':>8} {'delta':>8}")
print("-" * 34)
for ch, tp, wp in zip(channels, torch_penalty, ws_penalty):
    print(f"{ch:4.0f} {tp:8.1f} {wp:8.1f} {wp-tp:+8.1f}")
