#!/usr/bin/env python3
"""Generate publication-quality counter analysis plots for 8K GEMM (alone vs RCCL overlap)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DATA = "results/counters_8k_v2/counter_summary_v2.json"
OUT_DIR = "results/plots"

with open(DATA) as f:
    d = json.load(f)

def g(scenario, ktype, counter):
    return d.get(scenario, {}).get(ktype, {}).get(counter, {}).get("median", 0)

# ---------------------------------------------------------------------------
# Extract all metrics
# ---------------------------------------------------------------------------
torch_alone_l2_hit  = g("torch_alone", "GEMM", "TCC_HIT_sum")
torch_alone_l2_miss = g("torch_alone", "GEMM", "TCC_MISS_sum")
torch_rccl_l2_hit   = g("torch_rccl",  "GEMM", "TCC_HIT_sum")
torch_rccl_l2_miss  = g("torch_rccl",  "GEMM", "TCC_MISS_sum")
ws_alone_l2_hit     = g("ws_alone",    "GEMM", "TCC_HIT_sum")
ws_alone_l2_miss    = g("ws_alone",    "GEMM", "TCC_MISS_sum")
ws_rccl_l2_hit      = g("ws_rccl",     "GEMM", "TCC_HIT_sum")
ws_rccl_l2_miss     = g("ws_rccl",     "GEMM", "TCC_MISS_sum")

def hitrate(h, m): return h / (h + m) * 100 if (h + m) > 0 else 0

l2_rates = [
    hitrate(torch_alone_l2_hit, torch_alone_l2_miss),
    hitrate(torch_rccl_l2_hit,  torch_rccl_l2_miss),
    hitrate(ws_alone_l2_hit,    ws_alone_l2_miss),
    hitrate(ws_rccl_l2_hit,     ws_rccl_l2_miss),
]

mall_rates = [
    g("torch_alone", "GEMM", "MALL_HIT_RATE_1"),
    g("torch_rccl",  "GEMM", "MALL_HIT_RATE_1"),
    g("ws_alone",    "GEMM", "MALL_HIT_RATE_1"),
    g("ws_rccl",     "GEMM", "MALL_HIT_RATE_1"),
]

mall_bw_gemm = [
    g("torch_alone", "GEMM", "MALL_BANDWIDTH_ALL"),
    g("torch_rccl",  "GEMM", "MALL_BANDWIDTH_ALL"),
    g("ws_alone",    "GEMM", "MALL_BANDWIDTH_ALL"),
    g("ws_rccl",     "GEMM", "MALL_BANDWIDTH_ALL"),
]

mall_bw_rccl = [
    0,
    g("torch_rccl", "RCCL", "MALL_BANDWIDTH_ALL"),
    0,
    g("ws_rccl",    "RCCL", "MALL_BANDWIDTH_ALL"),
]

hbm_rd_gemm = [g(s, "GEMM", "HBM_READ_BYTES") / 1e6 for s in
               ["torch_alone", "torch_rccl", "ws_alone", "ws_rccl"]]
hbm_wr_gemm = [g(s, "GEMM", "HBM_WRITE_BYTES") / 1e6 for s in
               ["torch_alone", "torch_rccl", "ws_alone", "ws_rccl"]]
hbm_rd_rccl = [0, g("torch_rccl", "RCCL", "HBM_READ_BYTES") / 1e6,
               0, g("ws_rccl",    "RCCL", "HBM_READ_BYTES") / 1e6]
hbm_wr_rccl = [0, g("torch_rccl", "RCCL", "HBM_WRITE_BYTES") / 1e6,
               0, g("ws_rccl",    "RCCL", "HBM_WRITE_BYTES") / 1e6]

rccl_mall_rate = [
    g("torch_rccl", "RCCL", "MALL_HIT_RATE_1"),
    g("ws_rccl",    "RCCL", "MALL_HIT_RATE_1"),
]
rccl_l2_hit  = [g("torch_rccl", "RCCL", "TCC_HIT_sum"),
                g("ws_rccl",    "RCCL", "TCC_HIT_sum")]
rccl_l2_miss = [g("torch_rccl", "RCCL", "TCC_MISS_sum"),
                g("ws_rccl",    "RCCL", "TCC_MISS_sum")]
rccl_l2_rate = [hitrate(rccl_l2_hit[i], rccl_l2_miss[i]) for i in range(2)]

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
TORCH_COLOR   = "#4ADE80"
WS_COLOR      = "#FF6B8A"
RCCL_COLOR    = "#60A5FA"
ACCENT_YELLOW = "#FBBF24"
ACCENT_PURPLE = "#A78BFA"

ALONE_ALPHA   = 1.0
OVERLAP_ALPHA = 0.70
HATCH_OVERLAP = "///"

plt.rcParams.update({
    "figure.facecolor":  "#0D0D0D",
    "axes.facecolor":    "#141414",
    "axes.edgecolor":    "#444",
    "axes.labelcolor":   "#DDD",
    "text.color":        "#DDD",
    "xtick.color":       "#BBB",
    "ytick.color":       "#BBB",
    "grid.color":        "#333",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#1A1A1A",
    "legend.edgecolor":  "#555",
    "legend.labelcolor": "#DDD",
    "font.family":       "sans-serif",
    "font.size":         12,
})

labels = ["torch.matmul\nAlone", "torch.matmul\n+ RCCL",
          "WS Hierarchical\nAlone", "WS Hierarchical\n+ RCCL"]
x = np.arange(4)

# ===========================================================================
# PLOT 1 — L2 & MALL Hit Rates  (grouped bars, side-by-side)
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

colors = [TORCH_COLOR, TORCH_COLOR, WS_COLOR, WS_COLOR]
alphas = [ALONE_ALPHA, OVERLAP_ALPHA, ALONE_ALPHA, OVERLAP_ALPHA]
hatches = ["", HATCH_OVERLAP, "", HATCH_OVERLAP]
edge_colors = ["#333"] * 4

for ax, rates, title, ylbl in [
    (axes[0], l2_rates,   "L2 (TCC) Hit Rate",    "Hit Rate (%)"),
    (axes[1], mall_rates, "MALL (LLC) Hit Rate",   "Hit Rate (%)"),
]:
    bars = ax.bar(x, rates, width=0.55, color=colors,
                  edgecolor=edge_colors, linewidth=1.2)
    for b, a, h in zip(bars, alphas, hatches):
        b.set_alpha(a)
        b.set_hatch(h)

    for i, v in enumerate(rates):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#FFF")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylbl, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3)
    ymin = min(rates) - 3
    ymax = max(rates) + 3
    ax.set_ylim(max(0, ymin), min(100, ymax))

fig.suptitle("8192x8192 BF16 GEMM  —  Cache Hit Rates (Alone vs. RCCL Overlap)",
             fontsize=16, fontweight="bold", y=0.98, color="#FFF")
fig.tight_layout(rect=[0, 0, 1, 0.93])
out1 = f"{OUT_DIR}/counter_hitrates.png"
fig.savefig(out1, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out1}")

# ===========================================================================
# PLOT 2 — MALL Bandwidth (GEMM + RCCL stacked)
# ===========================================================================
fig, ax = plt.subplots(figsize=(12, 7))
w = 0.45

gemm_bw_k = [v / 1e3 for v in mall_bw_gemm]
rccl_bw_k = [v / 1e3 for v in mall_bw_rccl]

bars_g = ax.bar(x, gemm_bw_k, w, color=colors,
                edgecolor="#333", linewidth=1.2, label="GEMM")
for b, a, h in zip(bars_g, alphas, hatches):
    b.set_alpha(a); b.set_hatch(h)

rccl_alphas = [0, 0.8, 0, 0.8]
bars_r = ax.bar(x, rccl_bw_k, w, bottom=gemm_bw_k,
                color=[RCCL_COLOR]*4,
                edgecolor="#333", linewidth=1.2, label="RCCL")
for b, a in zip(bars_r, rccl_alphas):
    b.set_alpha(a)

for i in range(4):
    total = gemm_bw_k[i] + rccl_bw_k[i]
    ax.text(i, total + 20, f"{total:.0f}K", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#FFF")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("MALL Bandwidth (K units)", fontsize=13)
ax.set_title("MALL (LLC) Bandwidth  —  Data Fabric Traffic During Kernel",
             fontsize=15, fontweight="bold", pad=12)
ax.grid(axis="y", alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=TORCH_COLOR, edgecolor="#333", label="GEMM (torch.matmul)"),
    Patch(facecolor=WS_COLOR, edgecolor="#333", label="GEMM (WS Hierarchical)"),
    Patch(facecolor=RCCL_COLOR, alpha=0.8, edgecolor="#333", label="RCCL Traffic"),
    Patch(facecolor="#666", hatch=HATCH_OVERLAP, edgecolor="#333", label="+ RCCL Overlap"),
]
ax.legend(handles=legend_elements, fontsize=11, loc="upper left")
fig.tight_layout()
out2 = f"{OUT_DIR}/counter_mall_bandwidth.png"
fig.savefig(out2, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out2}")

# ===========================================================================
# PLOT 3 — HBM Traffic (stacked: GEMM read/write + RCCL read/write)
# ===========================================================================
fig, ax = plt.subplots(figsize=(14, 7))
w = 0.55

bottom = np.zeros(4)
bar_sets = []

b1 = ax.bar(x, hbm_rd_gemm, w, bottom=bottom, color=ACCENT_YELLOW, alpha=0.9,
            edgecolor="#333", linewidth=1.0, label="GEMM HBM Read")
bottom += hbm_rd_gemm

b2 = ax.bar(x, hbm_wr_gemm, w, bottom=bottom, color=ACCENT_PURPLE, alpha=0.9,
            edgecolor="#333", linewidth=1.0, label="GEMM HBM Write")
bottom += hbm_wr_gemm

b3 = ax.bar(x, hbm_rd_rccl, w, bottom=bottom, color=RCCL_COLOR, alpha=0.7,
            edgecolor="#333", linewidth=1.0, label="RCCL HBM Read")
bottom += hbm_rd_rccl

b4 = ax.bar(x, hbm_wr_rccl, w, bottom=bottom, color="#F472B6", alpha=0.7,
            edgecolor="#333", linewidth=1.0, label="RCCL HBM Write")
total_hbm = np.array(hbm_rd_gemm) + np.array(hbm_wr_gemm) + np.array(hbm_rd_rccl) + np.array(hbm_wr_rccl)

for i, v in enumerate(total_hbm):
    ax.text(i, v + 1, f"{v:.1f} MB", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#FFF")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("HBM Traffic (MB)", fontsize=13)
ax.set_title("HBM Read/Write Traffic  —  Per-Kernel Attribution",
             fontsize=15, fontweight="bold", pad=12)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=10, loc="upper left", ncol=2)
fig.tight_layout()
out3 = f"{OUT_DIR}/counter_hbm_traffic.png"
fig.savefig(out3, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out3}")

# ===========================================================================
# PLOT 4 — RCCL Kernel Cache Behavior (L2 + MALL side by side)
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
xlabels_r = ["torch + RCCL", "WS + RCCL"]
xr = np.arange(2)

ax = axes[0]
bars = ax.bar(xr, rccl_l2_rate, 0.45, color=[TORCH_COLOR, WS_COLOR],
              alpha=0.8, edgecolor="#333", linewidth=1.2)
for i, v in enumerate(rccl_l2_rate):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color="#FFF")
ax.set_xticks(xr)
ax.set_xticklabels(xlabels_r, fontsize=11)
ax.set_ylabel("Hit Rate (%)", fontsize=13)
ax.set_title("RCCL Kernel — L2 Hit Rate", fontsize=14, fontweight="bold", pad=10)
ax.set_ylim(0, 40)
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
bars = ax.bar(xr, rccl_mall_rate, 0.45, color=[TORCH_COLOR, WS_COLOR],
              alpha=0.8, edgecolor="#333", linewidth=1.2)
for i, v in enumerate(rccl_mall_rate):
    ax.text(i, v + 0.002, f"{v:.4f}%", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color="#FFF")
ax.set_xticks(xr)
ax.set_xticklabels(xlabels_r, fontsize=11)
ax.set_ylabel("Hit Rate (%)", fontsize=13)
ax.set_title("RCCL Kernel — MALL (LLC) Hit Rate", fontsize=14, fontweight="bold", pad=10)
ax.set_ylim(0, max(rccl_mall_rate) * 3)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("RCCL Kernel Cache Behavior  —  Near-Zero LLC Reuse (Streaming Traffic)",
             fontsize=15, fontweight="bold", y=0.99, color="#FFF")
fig.tight_layout(rect=[0, 0, 1, 0.93])
out4 = f"{OUT_DIR}/counter_rccl_cache.png"
fig.savefig(out4, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out4}")

# ===========================================================================
# PLOT 5 — Summary dashboard (2x2) — PPT-friendly 16:9
# ===========================================================================
ppt_rc = {
    "figure.facecolor": "#000000",
    "axes.facecolor":   "#0A0A0A",
    "axes.edgecolor":   "#333",
}
with plt.rc_context(ppt_rc):
    fig, axes = plt.subplots(2, 2, figsize=(20, 11.25))

    # (0,0) L2 hit rate
    ax = axes[0, 0]
    bars = ax.bar(x, l2_rates, 0.55, color=colors,
                  edgecolor="#333", linewidth=1.2)
    for b, a, h in zip(bars, alphas, hatches):
        b.set_alpha(a); b.set_hatch(h)
    for i, v in enumerate(l2_rates):
        ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#FFF")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Hit Rate (%)", fontsize=13)
    ax.set_title("L2 (TCC) Hit Rate", fontsize=15, fontweight="bold")
    ax.set_ylim(min(l2_rates) - 3, min(100, max(l2_rates) + 3))
    ax.grid(axis="y", alpha=0.3)

    # (0,1) MALL hit rate
    ax = axes[0, 1]
    bars = ax.bar(x, mall_rates, 0.55, color=colors,
                  edgecolor="#333", linewidth=1.2)
    for b, a, h in zip(bars, alphas, hatches):
        b.set_alpha(a); b.set_hatch(h)
    for i, v in enumerate(mall_rates):
        ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#FFF")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Hit Rate (%)", fontsize=13)
    ax.set_title("MALL (LLC) Hit Rate", fontsize=15, fontweight="bold")
    ax.set_ylim(min(mall_rates) - 3, min(100, max(mall_rates) + 3))
    ax.grid(axis="y", alpha=0.3)

    # (1,0) MALL bandwidth stacked
    ax = axes[1, 0]
    bars_g = ax.bar(x, gemm_bw_k, 0.5, color=colors,
                    edgecolor="#333", linewidth=1.0)
    for b, a, h in zip(bars_g, alphas, hatches):
        b.set_alpha(a); b.set_hatch(h)
    bars_r = ax.bar(x, rccl_bw_k, 0.5, bottom=gemm_bw_k,
                    color=[RCCL_COLOR]*4,
                    edgecolor="#333", linewidth=1.0)
    for b, a in zip(bars_r, rccl_alphas):
        b.set_alpha(a)
    for i in range(4):
        total = gemm_bw_k[i] + rccl_bw_k[i]
        ax.text(i, total + 15, f"{total:.0f}K", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#FFF")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("MALL BW (K units)", fontsize=13)
    ax.set_title("MALL Bandwidth (Data Fabric Traffic)", fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    legend_elements = [
        Patch(facecolor=TORCH_COLOR, edgecolor="#333", label="GEMM (torch)"),
        Patch(facecolor=WS_COLOR, edgecolor="#333", label="GEMM (WS)"),
        Patch(facecolor=RCCL_COLOR, alpha=0.8, edgecolor="#333", label="RCCL"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left")

    # (1,1) HBM traffic stacked
    ax = axes[1, 1]
    bottom = np.zeros(4)
    ax.bar(x, hbm_rd_gemm, 0.5, bottom=bottom, color=ACCENT_YELLOW, alpha=0.9,
           edgecolor="#333", linewidth=1.0, label="GEMM Read")
    bottom += hbm_rd_gemm
    ax.bar(x, hbm_wr_gemm, 0.5, bottom=bottom, color=ACCENT_PURPLE, alpha=0.9,
           edgecolor="#333", linewidth=1.0, label="GEMM Write")
    bottom += hbm_wr_gemm
    ax.bar(x, hbm_rd_rccl, 0.5, bottom=bottom, color=RCCL_COLOR, alpha=0.7,
           edgecolor="#333", linewidth=1.0, label="RCCL Read")
    bottom += hbm_rd_rccl
    ax.bar(x, hbm_wr_rccl, 0.5, bottom=bottom, color="#F472B6", alpha=0.7,
           edgecolor="#333", linewidth=1.0, label="RCCL Write")
    total = np.array(hbm_rd_gemm) + np.array(hbm_wr_gemm) + np.array(hbm_rd_rccl) + np.array(hbm_wr_rccl)
    for i, v in enumerate(total):
        ax.text(i, v + 1, f"{v:.0f} MB", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#FFF")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("HBM Traffic (MB)", fontsize=13)
    ax.set_title("HBM Traffic (Per-Kernel Attribution)", fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="upper left", ncol=2)

    fig.tight_layout(pad=1.5)
    out5 = f"{OUT_DIR}/counter_dashboard.png"
    fig.savefig(out5, dpi=200, bbox_inches="tight", facecolor="#000000")
    plt.close(fig)
    print(f"Saved {out5}")

print("\nAll plots generated.")
