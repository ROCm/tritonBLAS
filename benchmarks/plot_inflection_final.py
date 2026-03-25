#!/usr/bin/env python3
"""Single definitive inflection-point plot: penalty + wall time + verdict."""
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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

BG = "#0D1117"
CARD = "#161B22"
BORDER = "#30363D"
TEXT = "#C9D1D9"
DIM = "#8B949E"
TORCH_COLOR = "#FF6B6B"
WS_COLOR = "#4ECDC4"
COMM_COLOR = "#F7DC6F"
WIN_GREEN = "#2EA043"
LOSE_RED = "#DA3633"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": DIM,
    "ytick.color": DIM,
    "axes.edgecolor": BORDER,
    "grid.color": "#21262D",
    "grid.alpha": 0.6,
})

os.makedirs("results/plots", exist_ok=True)

files = [
    ("4K",  "results/overlap_data/inflection_overlap_4k.json"),
    ("8K",  "results/overlap_data/inflection_overlap_8k.json"),
    ("12K", "results/overlap_data/inflection_overlap.json"),
    ("16K", "results/overlap_data/inflection_overlap_16k.json"),
]

labels, torch_pen, ws_pen = [], [], []
torch_wall, ws_wall, ws_best_cu = [], [], []
torch_alone_ms, ws_alone_ms = [], []
comm_ms_list = []

for label, path in files:
    with open(path) as f:
        d = json.load(f)
    labels.append(label)
    torch_pen.append(d["torch"]["penalty_pct"])
    torch_wall.append(d["torch"]["overlap_wall_ms"])
    torch_alone_ms.append(d["torch"]["alone_ms"])
    comm_ms_list.append(d["comm_alone_ms"])

    best_key = min(d["ws_hierarchical"],
                   key=lambda k: d["ws_hierarchical"][k]["overlap_wall_ms"])
    best = d["ws_hierarchical"][best_key]
    ws_pen.append(best["penalty_pct"])
    ws_wall.append(best["overlap_wall_ms"])
    ws_alone_ms.append(best["alone_ms"])
    ws_best_cu.append(int(best_key))

# Wave-quantization last-wave occupancy at 304 CUs
wq = {"4K": 84.2, "8K": 36.8, "12K": 57.9, "16K": 47.4}

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                      left=0.08, right=0.96, top=0.90, bottom=0.08)

# ──────────────────────────────────────────────────────────────
# Panel A  (top-left): Overlap Penalty %
# ──────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(labels))
w = 0.32

bars_t = ax1.bar(x - w/2, torch_pen, w, color=TORCH_COLOR, edgecolor="none",
                 label="torch.matmul", zorder=3)
bars_w = ax1.bar(x + w/2, ws_pen, w, color=WS_COLOR, edgecolor="none",
                 label="WS Hierarchical", zorder=3)

for bar, val in zip(bars_t, torch_pen):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=10,
             color=TORCH_COLOR, fontweight="bold")
for bar, val in zip(bars_w, ws_pen):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=10,
             color=WS_COLOR, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=12)
ax1.set_ylabel("RCCL Overlap Penalty (%)", fontsize=12)
ax1.set_title("(a)  Overlap Penalty", fontsize=13, color=TEXT, fontweight="bold",
              pad=10)
ax1.legend(fontsize=10, loc="upper right",
           facecolor=CARD, edgecolor=BORDER)
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.set_ylim(0, max(torch_pen) * 1.3)

# ──────────────────────────────────────────────────────────────
# Panel B  (top-right): Overlap Wall Time (ms)
# ──────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

bars_tw = ax2.bar(x - w/2, torch_wall, w, color=TORCH_COLOR, edgecolor="none",
                  label="torch.matmul", zorder=3)
bars_ww = ax2.bar(x + w/2, ws_wall, w, color=WS_COLOR, edgecolor="none",
                  label="WS Hierarchical", zorder=3)

for i in range(len(labels)):
    diff_pct = (ws_wall[i] - torch_wall[i]) / torch_wall[i] * 100
    color = WIN_GREEN if diff_pct < 0 else LOSE_RED
    tag = f"WS {diff_pct:+.1f}%"
    ypos = max(torch_wall[i], ws_wall[i]) + 0.02 * max(max(torch_wall), max(ws_wall))
    ax2.text(x[i], ypos, tag, ha="center", va="bottom", fontsize=10,
             color=color, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=12)
ax2.set_ylabel("Overlap Wall Time (ms)", fontsize=12)
ax2.set_title("(b)  End-to-End Wall Time", fontsize=13, color=TEXT,
              fontweight="bold", pad=10)
ax2.legend(fontsize=10, loc="upper left",
           facecolor=CARD, edgecolor=BORDER)
ax2.grid(axis="y", alpha=0.3, zorder=0)

# ──────────────────────────────────────────────────────────────
# Panel C  (bottom-left): Raw GEMM alone + overlapped latency
# ──────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
bw = 0.18

ax3.bar(x - 1.5*bw, torch_alone_ms, bw, color=TORCH_COLOR, alpha=0.45,
        edgecolor=TORCH_COLOR, linewidth=0.8, label="torch alone", zorder=3)
ax3.bar(x - 0.5*bw, [d for d in torch_wall], bw, color=TORCH_COLOR,
        edgecolor="none", label="torch + RCCL", zorder=3)
ax3.bar(x + 0.5*bw, ws_alone_ms, bw, color=WS_COLOR, alpha=0.45,
        edgecolor=WS_COLOR, linewidth=0.8, label="WS alone", zorder=3)
ax3.bar(x + 1.5*bw, ws_wall, bw, color=WS_COLOR,
        edgecolor="none", label="WS + RCCL", zorder=3)

ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=12)
ax3.set_ylabel("Latency (ms)", fontsize=12)
ax3.set_title("(c)  Alone vs Overlapped Latency", fontsize=13, color=TEXT,
              fontweight="bold", pad=10)
ax3.legend(fontsize=9, loc="upper left", ncol=2,
           facecolor=CARD, edgecolor=BORDER)
ax3.grid(axis="y", alpha=0.3, zorder=0)

# ──────────────────────────────────────────────────────────────
# Panel D  (bottom-right): Wave-quant last-wave occupancy
# ──────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

occ = [wq[l] for l in labels]
colors_occ = [WIN_GREEN if ws_wall[i] < torch_wall[i] else LOSE_RED
              for i, l in enumerate(labels)]

bars_occ = ax4.bar(x, occ, 0.5, color=colors_occ, edgecolor="none", zorder=3)
ax4.axhline(y=50, color=DIM, linestyle="--", linewidth=1, alpha=0.5)
ax4.text(len(labels) - 0.5, 51, "50% threshold", fontsize=9, color=DIM,
         ha="right", va="bottom")

for i, (bar, val) in enumerate(zip(bars_occ, occ)):
    ws_wins = ws_wall[i] < torch_wall[i]
    verdict = "WS WINS" if ws_wins else "torch wins"
    v_color = WIN_GREEN if ws_wins else LOSE_RED
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{val:.0f}%\n{verdict}", ha="center", va="bottom",
             fontsize=10, color=v_color, fontweight="bold")

ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=12)
ax4.set_ylabel("Last-Wave Occupancy at 304 CUs (%)", fontsize=12)
ax4.set_title("(d)  Wave Quantization Severity → Verdict", fontsize=13,
              color=TEXT, fontweight="bold", pad=10)
ax4.grid(axis="y", alpha=0.3, zorder=0)
ax4.set_ylim(0, 110)

fig.suptitle("Inflection Point: Where Work-Stealing Beats torch.matmul Under RCCL Overlap",
             fontsize=16, color=TEXT, fontweight="bold", y=0.97)

fig.savefig("results/plots/inflection_final.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: results/plots/inflection_final.png")
