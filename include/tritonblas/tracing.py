# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Persistent GEMM trace visualization.

Produces a per-PID timeline PNG showing when each workgroup starts and
finishes every tile it processes. Tiles are color-coded by XCD (chiplet).
"""

import numpy as np


def plot_gemm_trace(trace_data, output_path="gemm_trace.png", title=None):
    """
    Render a persistent GEMM tile trace as a Gantt-style timeline PNG.

    Each row is a program ID (pid / workgroup). Horizontal bars show the
    wall-clock interval each tile occupied, color-coded by XCD.

    Args:
        trace_data: dict returned by persistent_matmul_lt(..., trace=True).
            Required keys: "start", "end", "pid", "xcd".
            Optional metadata keys: "M", "N", "K", "BLOCK_SIZE_M", etc.
        output_path: File path for the output PNG (default: "gemm_trace.png").
        title: Optional custom title for the plot.

    Returns:
        The output_path string.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    start_np = trace_data["start"].numpy().astype(np.int64)
    end_np = trace_data["end"].numpy().astype(np.int64)
    pids_np = trace_data["pid"].numpy().astype(np.int32)
    xcds_np = trace_data["xcd"].numpy().astype(np.int32)

    # Filter out tiles that were never written (both start and end are 0)
    valid = (start_np != 0) | (end_np != 0)
    start_np = start_np[valid]
    end_np = end_np[valid]
    pids_np = pids_np[valid]
    xcds_np = xcds_np[valid]

    if len(start_np) == 0:
        raise ValueError("Trace data contains no valid events.")

    # Normalize timestamps: convert to microseconds relative to earliest start
    # s_memrealtime runs at 100 MHz → 1 tick = 10 ns = 0.01 µs
    min_ts = start_np.min()
    start_us = (start_np - min_ts) * 0.01
    end_us = (end_np - min_ts) * 0.01

    unique_pids = np.unique(pids_np)
    num_pids = len(unique_pids)
    pid_to_y = {int(pid): i for i, pid in enumerate(unique_pids)}

    unique_xcds = np.unique(xcds_np)
    cmap = plt.cm.get_cmap("tab10", max(len(unique_xcds), 1))
    xcd_to_color = {int(xcd): cmap(i) for i, xcd in enumerate(unique_xcds)}

    fig_height = max(4, num_pids * 0.12 + 1.5)
    fig, ax = plt.subplots(figsize=(20, fig_height))

    bar_height = 0.8
    for i in range(len(start_us)):
        pid = int(pids_np[i])
        xcd = int(xcds_np[i])
        y = pid_to_y[pid]
        width = end_us[i] - start_us[i]
        ax.barh(
            y, width, left=start_us[i], height=bar_height,
            color=xcd_to_color[xcd], edgecolor="black", linewidth=0.3,
        )

    ax.set_xlim(0, end_us.max() * 1.02)
    ax.set_ylim(-0.5, num_pids - 0.5)
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Program ID (pid)")

    # Build title with metadata if available
    if title is None:
        parts = ["Persistent GEMM Tile Trace"]
        M = trace_data.get("M")
        N = trace_data.get("N")
        K = trace_data.get("K")
        if M is not None:
            parts.append(f"M={M} N={N} K={K}")
        bm = trace_data.get("BLOCK_SIZE_M")
        bn = trace_data.get("BLOCK_SIZE_N")
        if bm is not None:
            parts.append(f"BLK={bm}x{bn}")
        tt = trace_data.get("total_tiles")
        tp = trace_data.get("total_programs")
        if tt is not None:
            parts.append(f"tiles={tt} pids={tp}")
        title = "  |  ".join(parts)
    ax.set_title(title)

    # Only show a subset of y-tick labels when there are many pids
    if num_pids > 64:
        step = max(1, num_pids // 32)
        tick_positions = list(range(0, num_pids, step))
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([unique_pids[i] for i in tick_positions], fontsize=6)
    else:
        ax.set_yticks(range(num_pids))
        ax.set_yticklabels(unique_pids, fontsize=max(4, 8 - num_pids // 20))

    legend_handles = [
        mpatches.Patch(color=xcd_to_color[xcd], label=f"XCD {xcd}")
        for xcd in unique_xcds
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    total_us = end_us.max()
    print(f"Trace saved to {output_path}")
    print(f"  {len(start_us)} tiles across {num_pids} pids, "
          f"total span {total_us:.1f} µs")

    return output_path
