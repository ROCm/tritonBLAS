#!/usr/bin/env python3
"""Generate all overlap analysis plots.

Black background, pastel colors, consistent styling.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUT = "results/plots"
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

WS = "#FF6B8A"
TORCH = "#6BB5FF"
STREAMK = "#FFB86B"
IDEAL = "#7BFF7B"
WS_LIGHT = "#FF9EB5"
TORCH_LIGHT = "#9ED4FF"

# Load data
try:
    with open("results/plot_data/rotating_overlap.json") as f:
        ovl = json.load(f)
except FileNotFoundError:
    with open("results/plot_data/overlap_optimal_cpx.json") as f:
        ovl = json.load(f)
with open("results/plot_data/cu_sweep_8192_v2.json") as f:
    cu8k = json.load(f)
with open("results/plot_data/cu_sweep_4096_v2.json") as f:
    cu4k = json.load(f)
with open("results/plot_data/distribution_8k.json") as f:
    dist8k = json.load(f)

SIZES = [1024, 2048, 4096, 8192, 12288, 16384]
SLABELS = ["1K", "2K", "4K", "8K", "12K", "16K"]

def med(d, key):
    return d.get(key + "_median", 0)

# Mask-bits to approximate CU count mapping for MI300X
# From empirical data: steps at bits 8, 16, 24, 32 correspond to
# enabling 1, 2, 3, 4 "CU lanes" across all SEs
# At 38 bits (max), we enable all CU lanes
# Map linearly: actual_cus = bits * 8 for bits <= 38
def mask_bits_to_cus(bits):
    return min(bits * 8, 304)


# =====================================================================
# PLOT 1: Overlap penalty bars
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(SIZES))
width = 0.35

ws_pen = [(med(ovl[f"ws_{sz}"], "overlap") / med(ovl[f"ws_{sz}"], "alone") - 1) * 100
          for sz in SIZES]
to_pen = [(med(ovl[f"torch_{sz}"], "overlap") / med(ovl[f"torch_{sz}"], "alone") - 1) * 100
          for sz in SIZES]

b1 = ax.bar(x - width/2, ws_pen, width, label="Work-Stealing", color=WS, edgecolor="#0D0D0D", lw=0.5)
b2 = ax.bar(x + width/2, to_pen, width, label="torch.matmul", color=TORCH, edgecolor="#0D0D0D", lw=0.5)
ax.axhline(y=10.5, color=IDEAL, ls="--", lw=1.5, alpha=0.7, label="Ideal (-32 CUs = 10.5%)")

for bar, val in zip(b1, ws_pen):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=9, color=WS, fontweight="bold")
for bar, val in zip(b2, to_pen):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=9, color=TORCH, fontweight="bold")

ax.set_xlabel("GEMM Size (M=N=K)", fontsize=13)
ax.set_ylabel("RCCL Overlap Penalty (%)", fontsize=13)
ax.set_title("RCCL Overlap Penalty: WS vs torch.matmul\n8-GPU all_reduce, BF16, MI300X", fontsize=15, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(SLABELS, fontsize=12)
ax.legend(fontsize=11, loc="upper right"); ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(max(to_pen), max(ws_pen)) * 1.25)
fig.tight_layout(); fig.savefig(f"{OUT}/1_overlap_penalty_bars.png", dpi=200)
print(f"Saved 1_overlap_penalty_bars.png"); plt.close()

# =====================================================================
# PLOT 1b: Latency bars (ms)
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 7))
width = 0.2
x = np.arange(len(SIZES))

ws_alone = [med(ovl[f"ws_{sz}"], "alone") for sz in SIZES]
ws_ovl = [med(ovl[f"ws_{sz}"], "overlap") for sz in SIZES]
to_alone = [med(ovl[f"torch_{sz}"], "alone") for sz in SIZES]
to_ovl = [med(ovl[f"torch_{sz}"], "overlap") for sz in SIZES]

ax.bar(x - 1.5*width, ws_alone, width, label="WS alone", color=WS, alpha=0.7, edgecolor="#0D0D0D")
ax.bar(x - 0.5*width, ws_ovl, width, label="WS + RCCL", color=WS, edgecolor="#0D0D0D")
ax.bar(x + 0.5*width, to_alone, width, label="torch alone", color=TORCH, alpha=0.7, edgecolor="#0D0D0D")
ax.bar(x + 1.5*width, to_ovl, width, label="torch + RCCL", color=TORCH, edgecolor="#0D0D0D")

ax.set_xlabel("GEMM Size (M=N=K)", fontsize=13)
ax.set_ylabel("Kernel Latency (ms)", fontsize=13)
ax.set_title("GEMM Latency: Alone vs RCCL Overlap\nWS vs torch.matmul, 8-GPU all_reduce, BF16, MI300X", fontsize=15, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(SLABELS, fontsize=12)
ax.set_yscale("log"); ax.legend(fontsize=10, ncol=2); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/1b_latency_bars.png", dpi=200)
print(f"Saved 1b_latency_bars.png"); plt.close()

# =====================================================================
# PLOT 2: Overlap penalty curve
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 7))
prob = [2 * sz**3 for sz in SIZES]
ax.plot(prob, ws_pen, "o-", color=WS, lw=2.5, ms=8, label="Work-Stealing")
ax.plot(prob, to_pen, "s-", color=TORCH, lw=2.5, ms=8, label="torch.matmul")
ax.axhline(y=10.5, color=IDEAL, ls="--", lw=1.5, alpha=0.7, label="Ideal CU loss (10.5%)")
for i, sz in enumerate(SIZES):
    ax.annotate(SLABELS[i], (prob[i], ws_pen[i]), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=9, color=WS)
ax.set_xscale("log")
ax.set_xlabel("Problem Size (FLOPs = 2*M*N*K)", fontsize=13)
ax.set_ylabel("RCCL Overlap Penalty (%)", fontsize=13)
ax.set_title("Overlap Penalty vs Problem Size\n8-GPU all_reduce, BF16, MI300X", fontsize=15, fontweight="bold")
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/2_overlap_penalty_curve.png", dpi=200)
print(f"Saved 2_overlap_penalty_curve.png"); plt.close()

# =====================================================================
# PLOT 3: 8K distribution (boxplot)
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 7))
cats = ["GEMM Alone\n(warm)", "GEMM Alone\n(rotating)", "Overlap GEMM\n(warm)", "Overlap GEMM\n(rotating)"]
d8ws = ovl.get("ws_8192", {})
d8to = ovl.get("torch_8192", {})
ws_d = [d8ws.get("alone_all", []), dist8k.get("ws_rotating", []),
        d8ws.get("overlap_all", []), []]
torch_d = [d8to.get("alone_all", []), dist8k.get("torch_rotating", []),
           d8to.get("overlap_all", []), []]

pos_ws = np.arange(len(cats)) * 2 - 0.3
pos_to = np.arange(len(cats)) * 2 + 0.3

bp_ws = ax.boxplot([d for d in ws_d if d], positions=[pos_ws[i] for i, d in enumerate(ws_d) if d],
                    widths=0.5, patch_artist=True, showfliers=False)
bp_to = ax.boxplot([d for d in torch_d if d], positions=[pos_to[i] for i, d in enumerate(torch_d) if d],
                    widths=0.5, patch_artist=True, showfliers=False)

for patch in bp_ws["boxes"]:
    patch.set_facecolor(WS); patch.set_alpha(0.6); patch.set_edgecolor(WS)
for patch in bp_to["boxes"]:
    patch.set_facecolor(TORCH); patch.set_alpha(0.6); patch.set_edgecolor(TORCH)
for el in ["whiskers", "caps"]:
    for l in bp_ws[el]: l.set_color(WS); l.set_alpha(0.7)
    for l in bp_to[el]: l.set_color(TORCH); l.set_alpha(0.7)
for l in bp_ws["medians"]: l.set_color("white"); l.set_linewidth(2)
for l in bp_to["medians"]: l.set_color("white"); l.set_linewidth(2)

ax.set_xticks(np.arange(len(cats)) * 2)
ax.set_xticklabels(cats, fontsize=11)
ax.set_ylabel("Kernel Time (ms)", fontsize=13)
ax.set_title("8192x8192x8192 BF16 - Per-Iteration Distribution\nWS (coral) vs torch.matmul (blue), MI300X",
             fontsize=14, fontweight="bold")
ax.legend(handles=[mpatches.Patch(facecolor=WS, alpha=0.6, label="Work-Stealing"),
                    mpatches.Patch(facecolor=TORCH, alpha=0.6, label="torch.matmul")], fontsize=11)
ax.grid(axis="y", alpha=0.3)
all_v = []
for ds in ws_d + torch_d:
    all_v.extend([v for v in ds if v < 10])
if all_v:
    ax.set_ylim(min(all_v) * 0.9, max(all_v) * 1.15)
fig.tight_layout(); fig.savefig(f"{OUT}/3_distribution_8k.png", dpi=200)
print(f"Saved 3_distribution_8k.png"); plt.close()

# =====================================================================
# PLOT 4: Cache counters
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
conditions = ["Alone\nWarm", "Alone\nRotating", "RCCL\nWarm", "RCCL\nRotating"]
ccolors = [WS, WS_LIGHT, TORCH, TORCH_LIGHT]

ax = axes[0]
hits = [78.22, 76.80, 78.51, 77.73]
bars = ax.bar(range(4), hits, color=ccolors, edgecolor="#0D0D0D", lw=0.5)
for b, v in zip(bars, hits):
    ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10, color="#DDD", fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(conditions, fontsize=9)
ax.set_ylabel("Hit Rate (%)"); ax.set_title("L2 (TCC) Hit Rate", fontsize=13, fontweight="bold")
ax.set_ylim(70, 82); ax.grid(axis="y", alpha=0.3)

ax = axes[1]
dram = [433.7, 449.9, 425.1, 438.0]
bars = ax.bar(range(4), dram, color=ccolors, edgecolor="#0D0D0D", lw=0.5)
for b, v in zip(bars, dram):
    ax.text(b.get_x() + b.get_width()/2, v + 3, f"{v:.0f}M", ha="center", fontsize=10, color="#DDD", fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(conditions, fontsize=9)
ax.set_ylabel("DRAM Read Requests (M)")
ax.set_title("DRAM Traffic (TCC_EA0_RDREQ)", fontsize=13, fontweight="bold")
ax.set_ylim(380, 480); ax.grid(axis="y", alpha=0.3)

ax = axes[2]
l1_labels = ["TCP_TOTAL\nACCESSES", "TCP_CACHE\nACCESSES", "TCP_TCC\nREAD_REQ", "TCP_TCC\nWRITE_REQ"]
l1_vals = [16612, 4530, 2013, 126]
x2 = np.arange(4)
ax.bar(x2 - 0.15, l1_vals, 0.3, color=WS, alpha=0.7, label="Warm")
ax.bar(x2 + 0.15, l1_vals, 0.3, color=WS_LIGHT, alpha=0.7, label="Rotating")
ax.set_xticks(x2); ax.set_xticklabels(l1_labels, fontsize=8)
ax.set_ylabel("Counter Value (M)")
ax.set_title("L1 (TCP) -- Identical", fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

fig.suptitle("Cache Counter Analysis: WS Kernel 8192x8192x8192 BF16, MI300X", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/4_cache_counters.png", dpi=200, bbox_inches="tight")
print(f"Saved 4_cache_counters.png"); plt.close()


# =====================================================================
# PLOT 5 & 6: CU sweep (8K and 4K)
# =====================================================================
def plot_cu_sweep(cu_data, sz, filename):
    fig, ax = plt.subplots(figsize=(14, 8))

    # WS grid-limited
    ws = cu_data.get("ws", [])
    ws_cus = [p["cus"] for p in ws if p.get("tflops")]
    ws_tf = [p["tflops"] for p in ws if p.get("tflops")]
    ax.plot(ws_cus, ws_tf, "o-", color=WS, lw=2.5, ms=5, label="WS (grid-limited)", zorder=3)

    # StreamK+WS
    sk = cu_data.get("streamk-ws", [])
    sk_cus = [p["cus"] for p in sk if p.get("tflops") and p["tflops"] > 50]
    sk_tf = [p["tflops"] for p in sk if p.get("tflops") and p["tflops"] > 50]
    if sk_cus:
        ax.plot(sk_cus, sk_tf, "D-", color=STREAMK, lw=2, ms=5, label="StreamK+WS (grid-limited)", zorder=3)

    # Torch CU-masked
    tc = cu_data.get("torch", cu_data.get("torch_masked", []))
    # Separate no-mask point
    tc_masked = [p for p in tc if p.get("mask", "none") != "none"]
    tc_nomask = [p for p in tc if p.get("mask", "none") == "none"]

    # Map mask bits to CU count (8 CUs per bit)
    tc_cus = [mask_bits_to_cus(p["cus"]) for p in tc_masked if p.get("tflops")]
    tc_tf = [p["tflops"] for p in tc_masked if p.get("tflops")]
    ax.plot(tc_cus, tc_tf, "s-", color=TORCH, lw=2, ms=4, alpha=0.8,
            label="torch.mm (CU-masked)", zorder=2)

    if tc_nomask:
        nm = tc_nomask[0]
        ax.scatter([304], [nm["tflops"]], color=TORCH, s=150, marker="*", zorder=5,
                   label=f"torch.mm (no mask): {nm['tflops']:.0f} TF")

    ax.set_xlabel("Available CUs", fontsize=13)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=13)
    ax.set_title(f"{sz}x{sz}x{sz} BF16 -- Available CUs vs Performance\nMI300X",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 320)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{filename}", dpi=200)
    print(f"Saved {filename}")
    plt.close()

plot_cu_sweep(cu8k, 8192, "5_cu_sweep_8k.png")
plot_cu_sweep(cu4k, 4096, "6_cu_sweep_4k.png")

# =====================================================================
# PLOT 7: TFLOPS comparison — alone vs during overlap
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

ws_alone_tf = [2.0 * sz**3 / (med(ovl[f"ws_{sz}"], "alone") * 1e-3) / 1e12 for sz in SIZES]
ws_ovl_tf = [2.0 * sz**3 / (med(ovl[f"ws_{sz}"], "overlap") * 1e-3) / 1e12 for sz in SIZES]
to_alone_tf = [2.0 * sz**3 / (med(ovl[f"torch_{sz}"], "alone") * 1e-3) / 1e12 for sz in SIZES]
to_ovl_tf = [2.0 * sz**3 / (med(ovl[f"torch_{sz}"], "overlap") * 1e-3) / 1e12 for sz in SIZES]

x = np.arange(len(SIZES))
width = 0.2

# Left: Alone TFLOPS
ax1.bar(x - width/2, ws_alone_tf, width, label="WS alone", color=WS, edgecolor="#0D0D0D")
ax1.bar(x + width/2, to_alone_tf, width, label="torch alone", color=TORCH, edgecolor="#0D0D0D")
ax1.set_xticks(x); ax1.set_xticklabels(SLABELS, fontsize=12)
ax1.set_ylabel("TFLOPS", fontsize=13)
ax1.set_title("GEMM Alone (rotating bufs)", fontsize=14, fontweight="bold")
ax1.legend(fontsize=11); ax1.grid(axis="y", alpha=0.3)

# Right: Overlap TFLOPS
ax2.bar(x - width/2, ws_ovl_tf, width, label="WS + RCCL", color=WS, edgecolor="#0D0D0D")
ax2.bar(x + width/2, to_ovl_tf, width, label="torch + RCCL", color=TORCH, edgecolor="#0D0D0D")
ax2.set_xticks(x); ax2.set_xticklabels(SLABELS, fontsize=12)
ax2.set_ylabel("TFLOPS", fontsize=13)
ax2.set_title("GEMM During RCCL Overlap (rotating bufs)", fontsize=14, fontweight="bold")
ax2.legend(fontsize=11); ax2.grid(axis="y", alpha=0.3)

# Match y-axes
ymax = max(max(ws_alone_tf + to_alone_tf), max(ws_ovl_tf + to_ovl_tf)) * 1.1
ax1.set_ylim(0, ymax); ax2.set_ylim(0, ymax)

# Annotate the 8K convergence
idx_8k = SIZES.index(8192)
ax2.annotate(f"WS={ws_ovl_tf[idx_8k]:.0f}\ntorch={to_ovl_tf[idx_8k]:.0f}",
             xy=(idx_8k, max(ws_ovl_tf[idx_8k], to_ovl_tf[idx_8k])),
             xytext=(idx_8k + 0.5, max(ws_ovl_tf[idx_8k], to_ovl_tf[idx_8k]) + 40),
             fontsize=10, color=IDEAL, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=IDEAL, lw=1.5))

fig.suptitle("Effective Compute: WS vs torch.matmul (rotating buffers)\n8-GPU all_reduce, BF16, MI300X",
             fontsize=15, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/7_tflops_comparison.png", dpi=200)
print(f"Saved 7_tflops_comparison.png"); plt.close()

print(f"\nAll plots saved to {OUT}/")
