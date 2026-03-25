#!/usr/bin/env python3
"""Generate all overlap analysis plots.

Black background, pastel colors, consistent styling.
Reads data from results/plot_data/ as collected by collect_all_data.py and cu_sweep.py.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

WS = "#FF6B8A"
TORCH = "#6BB5FF"
STREAMK = "#FFB86B"
PERSIST = "#B39DDB"
SK_WS = "#4DD0E1"
WS_FULL = "#FFAB91"
IDEAL = "#7BFF7B"
WS_LIGHT = "#FF9EB5"
TORCH_LIGHT = "#9ED4FF"

SIZES = [1024, 2048, 4096, 8192, 12288, 16384]
SLABELS = ["1K", "2K", "4K", "8K", "12K", "16K"]


# =====================================================================
# Data loading
# =====================================================================

def load_overlap_data():
    ovl = {}
    for backend in ["ws", "torch"]:
        for sz in SIZES:
            path = f"{DATA}/{backend}_{sz}.json"
            if os.path.exists(path):
                with open(path) as f:
                    ovl[f"{backend}_{sz}"] = json.load(f)
    if ovl:
        return ovl
    for fname in ["rotating_overlap.json", "overlap_optimal_cpx.json"]:
        path = f"{DATA}/{fname}"
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError("No overlap data found")


def med(d, key):
    if f"{key}_median" in d:
        return d[f"{key}_median"]
    if key in d and isinstance(d[key], dict):
        return d[key].get("median", 0)
    return 0


def load_cu_sweep(sz):
    for ver in ["v4", "v3", "v2", ""]:
        suffix = f"_{ver}" if ver else ""
        path = f"{DATA}/cu_sweep_{sz}{suffix}.json"
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}


def load_distribution():
    path = f"{DATA}/distribution_8k.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


ovl = load_overlap_data()
cu8k = load_cu_sweep(8192)
cu4k = load_cu_sweep(4096)
dist8k = load_distribution()


def get_alone_ms(backend, sz):
    key = f"{backend}_{sz}"
    if key not in ovl:
        return 0
    return med(ovl[key], "rotating") or med(ovl[key], "alone")


def get_overlap_ms(backend, sz):
    key = f"{backend}_{sz}"
    if key not in ovl:
        return 0
    return med(ovl[key], "rot_overlap_mm") or med(ovl[key], "overlap")


# =====================================================================
# PLOT 1: Overlap penalty bars
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(SIZES))
width = 0.35

ws_pen, to_pen = [], []
for sz in SIZES:
    a, o = get_alone_ms("ws", sz), get_overlap_ms("ws", sz)
    ws_pen.append((o / a - 1) * 100 if a > 0 else 0)
    a, o = get_alone_ms("torch", sz), get_overlap_ms("torch", sz)
    to_pen.append((o / a - 1) * 100 if a > 0 else 0)

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
ax.set_ylim(0, max(max(to_pen), max(ws_pen)) * 1.25 + 5)
fig.tight_layout(); fig.savefig(f"{OUT}/1_overlap_penalty_bars.png", dpi=200)
print("Saved 1_overlap_penalty_bars.png"); plt.close()


# =====================================================================
# PLOT 1b: Latency bars (ms)
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 7))
width = 0.2
x = np.arange(len(SIZES))

ws_alone = [get_alone_ms("ws", sz) for sz in SIZES]
ws_ovl = [get_overlap_ms("ws", sz) for sz in SIZES]
to_alone = [get_alone_ms("torch", sz) for sz in SIZES]
to_ovl = [get_overlap_ms("torch", sz) for sz in SIZES]

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
print("Saved 1b_latency_bars.png"); plt.close()


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
print("Saved 2_overlap_penalty_curve.png"); plt.close()


# =====================================================================
# PLOT 3: 8K distribution (boxplot)
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 7))
cats = ["Alone\n(warm)", "Alone\n(rotating)", "Overlap\n(warm)", "Overlap\n(rotating)"]

ws_d = [
    dist8k.get("ws_alone", ovl.get("ws_8192", {}).get("alone_all", [])),
    dist8k.get("ws_rotating", ovl.get("ws_8192", {}).get("rotating_all", [])),
    dist8k.get("ws_overlap", ovl.get("ws_8192", {}).get("overlap_all", [])),
    dist8k.get("ws_rot_overlap", ovl.get("ws_8192", {}).get("rot_overlap_mm_all", [])),
]
torch_d = [
    dist8k.get("torch_alone", ovl.get("torch_8192", {}).get("alone_all", [])),
    dist8k.get("torch_rotating", ovl.get("torch_8192", {}).get("rotating_all", [])),
    dist8k.get("torch_overlap", ovl.get("torch_8192", {}).get("overlap_all", [])),
    dist8k.get("torch_rot_overlap", ovl.get("torch_8192", {}).get("rot_overlap_mm_all", [])),
]

ws_nonempty = [(i, d) for i, d in enumerate(ws_d) if d]
torch_nonempty = [(i, d) for i, d in enumerate(torch_d) if d]

if ws_nonempty or torch_nonempty:
    pos_ws = np.arange(len(cats)) * 2 - 0.3
    pos_to = np.arange(len(cats)) * 2 + 0.3

    if ws_nonempty:
        bp_ws = ax.boxplot([d for _, d in ws_nonempty],
                           positions=[pos_ws[i] for i, _ in ws_nonempty],
                           widths=0.5, patch_artist=True, showfliers=False)
        for patch in bp_ws["boxes"]:
            patch.set_facecolor(WS); patch.set_alpha(0.6); patch.set_edgecolor(WS)
        for el in ["whiskers", "caps"]:
            for line in bp_ws[el]: line.set_color(WS); line.set_alpha(0.7)
        for line in bp_ws["medians"]: line.set_color("white"); line.set_linewidth(2)

    if torch_nonempty:
        bp_to = ax.boxplot([d for _, d in torch_nonempty],
                           positions=[pos_to[i] for i, _ in torch_nonempty],
                           widths=0.5, patch_artist=True, showfliers=False)
        for patch in bp_to["boxes"]:
            patch.set_facecolor(TORCH); patch.set_alpha(0.6); patch.set_edgecolor(TORCH)
        for el in ["whiskers", "caps"]:
            for line in bp_to[el]: line.set_color(TORCH); line.set_alpha(0.7)
        for line in bp_to["medians"]: line.set_color("white"); line.set_linewidth(2)

    ax.set_xticks(np.arange(len(cats)) * 2)
    ax.set_xticklabels(cats, fontsize=11)

ax.set_ylabel("Kernel Time (ms)", fontsize=13)
ax.set_title("8192x8192x8192 BF16 - Per-Iteration Distribution\nWS (coral) vs torch.matmul (blue), MI300X",
             fontsize=14, fontweight="bold")
ax.legend(handles=[mpatches.Patch(facecolor=WS, alpha=0.6, label="Work-Stealing"),
                    mpatches.Patch(facecolor=TORCH, alpha=0.6, label="torch.matmul")], fontsize=11)
ax.grid(axis="y", alpha=0.3)
all_v = [v for ds in ws_d + torch_d for v in ds if 0 < v < 20]
if all_v:
    ax.set_ylim(min(all_v) * 0.9, max(all_v) * 1.15)
fig.tight_layout(); fig.savefig(f"{OUT}/3_distribution_8k.png", dpi=200)
print("Saved 3_distribution_8k.png"); plt.close()


# =====================================================================
# PLOT 5 & 6: CU sweep (8K and 4K) — multi-backend
# =====================================================================

BACKEND_STYLE = {
    "torch":           {"color": TORCH,   "marker": "s", "lw": 2.5, "ms": 6, "label": "torch.matmul (CU-masked)"},
    "ws_grid":         {"color": WS,      "marker": "o", "lw": 2.5, "ms": 6, "label": "Work-Stealing (grid-limited)"},
    "ws_full":         {"color": WS_FULL, "marker": "^", "lw": 2,   "ms": 6, "label": "WS (full-grid, CU-masked)"},
    "persistent":      {"color": PERSIST, "marker": "D", "lw": 2,   "ms": 5, "label": "Persistent GEMM (CU-masked)"},
    "streamk":         {"color": STREAMK, "marker": "v", "lw": 2,   "ms": 5, "label": "Stream-K (CU-masked)"},
    "streamk_ws":      {"color": SK_WS,   "marker": "P", "lw": 2,   "ms": 6, "label": "SK+WS (CU-masked)"},
    "streamk_ws_grid":       {"color": "#00E5FF","marker": "D", "lw": 2.5, "ms": 7, "label": "Stream-K + WS (grid-limited)"},
    "streamk_ws_sk_enabled": {"color": "#00E5FF","marker": "h", "lw": 2,   "ms": 6, "label": "SK+WS grid-lim (SK on)", "alpha": 0.5},
    "streamk_static":        {"color": STREAMK, "marker": "*", "lw": 2.5, "ms": 8, "label": "Static Stream-K (grid-limited)"},
    "ws_global_atomic":      {"color": "#00E5FF","marker": "D", "lw": 2.5, "ms": 7, "label": "WS Global-Atomic (grid-limited)"},
    # Legacy keys
    "ws":              {"color": WS,      "marker": "o", "lw": 2.5, "ms": 6, "label": "WS (grid-limited)"},
}


def plot_cu_sweep(cu_data, sz, filename):
    if not cu_data:
        print(f"  Skipping {filename} (no data)")
        return

    fig, ax = plt.subplots(figsize=(16, 9))

    all_cus = set()

    for key in ["torch", "ws_grid", "streamk_static", "ws_global_atomic"]:
        series = cu_data.get(key, [])
        if not series:
            continue
        style = BACKEND_STYLE.get(key, {"color": "#888", "marker": ".", "lw": 1, "ms": 4, "label": key})
        pts = sorted(series, key=lambda p: p["cus"])
        cus = [p["cus"] for p in pts if p.get("tflops") and p["tflops"] > 0]
        tf = [p["tflops"] for p in pts if p.get("tflops") and p["tflops"] > 0]
        if cus:
            ax.plot(cus, tf, f"{style['marker']}-",
                    color=style["color"], lw=style["lw"], ms=style["ms"],
                    label=style["label"], zorder=3,
                    alpha=style.get("alpha", 1.0))
            all_cus.update(cus)

    # X-axis: ticks at actual CU values (mask increments: 8 per XCD)
    if all_cus:
        sorted_cus = sorted(all_cus)
        # Use every other value if too dense
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

    n_tiles = {"4096": 256, "8192": 1024, "12288": 2304, "16384": 4096}
    tiles = n_tiles.get(str(sz), "?")
    ax.set_title(f"{sz}x{sz}x{sz} BF16",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
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

ws_alone_tf = [2.0 * sz**3 / (get_alone_ms("ws", sz) * 1e-3) / 1e12
               if get_alone_ms("ws", sz) > 0 else 0 for sz in SIZES]
ws_ovl_tf = [2.0 * sz**3 / (get_overlap_ms("ws", sz) * 1e-3) / 1e12
             if get_overlap_ms("ws", sz) > 0 else 0 for sz in SIZES]
to_alone_tf = [2.0 * sz**3 / (get_alone_ms("torch", sz) * 1e-3) / 1e12
               if get_alone_ms("torch", sz) > 0 else 0 for sz in SIZES]
to_ovl_tf = [2.0 * sz**3 / (get_overlap_ms("torch", sz) * 1e-3) / 1e12
             if get_overlap_ms("torch", sz) > 0 else 0 for sz in SIZES]

x = np.arange(len(SIZES))
width = 0.2

ax1.bar(x - width/2, ws_alone_tf, width, label="WS alone", color=WS, edgecolor="#0D0D0D")
ax1.bar(x + width/2, to_alone_tf, width, label="torch alone", color=TORCH, edgecolor="#0D0D0D")
ax1.set_xticks(x); ax1.set_xticklabels(SLABELS, fontsize=12)
ax1.set_ylabel("TFLOPS", fontsize=13)
ax1.set_title("GEMM Alone (rotating bufs)", fontsize=14, fontweight="bold")
ax1.legend(fontsize=11); ax1.grid(axis="y", alpha=0.3)

ax2.bar(x - width/2, ws_ovl_tf, width, label="WS + RCCL", color=WS, edgecolor="#0D0D0D")
ax2.bar(x + width/2, to_ovl_tf, width, label="torch + RCCL", color=TORCH, edgecolor="#0D0D0D")
ax2.set_xticks(x); ax2.set_xticklabels(SLABELS, fontsize=12)
ax2.set_ylabel("TFLOPS", fontsize=13)
ax2.set_title("GEMM During RCCL Overlap (rotating bufs)", fontsize=14, fontweight="bold")
ax2.legend(fontsize=11); ax2.grid(axis="y", alpha=0.3)

ymax = max(max(ws_alone_tf + to_alone_tf), max(ws_ovl_tf + to_ovl_tf)) * 1.1
ax1.set_ylim(0, ymax); ax2.set_ylim(0, ymax)

idx_8k = SIZES.index(8192)
if ws_ovl_tf[idx_8k] > 0 and to_ovl_tf[idx_8k] > 0:
    ax2.annotate(f"WS={ws_ovl_tf[idx_8k]:.0f}\ntorch={to_ovl_tf[idx_8k]:.0f}",
                 xy=(idx_8k, max(ws_ovl_tf[idx_8k], to_ovl_tf[idx_8k])),
                 xytext=(idx_8k + 0.5, max(ws_ovl_tf[idx_8k], to_ovl_tf[idx_8k]) + 40),
                 fontsize=10, color=IDEAL, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=IDEAL, lw=1.5))

fig.suptitle("Effective Compute: WS vs torch.matmul (rotating buffers)\n8-GPU all_reduce, BF16, MI300X",
             fontsize=15, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/7_tflops_comparison.png", dpi=200)
print("Saved 7_tflops_comparison.png"); plt.close()

print(f"\nAll plots saved to {OUT}/")
