# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Let a tritonBLAS GEMM asynchronously wake a waiting Triton kernel."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.hip import memrealtime

import tritonblas


@triton.jit
def observe_tile_triggers(
    gates,
    signal_of_tile,
    signal_producers,
    output,
    observation_order,
    observation_counter,
    observed,
    observer_bounds,
    num_tiles,
    num_observers,
    tiles_n,
    block_m,
    block_n,
    output_stride,
    gate_stride: tl.constexpr,
    gate_base: tl.constexpr,
):
    """
    Order and consume every tile as its GEMM gate opens.

    A bounded observer count reduces scan-order bias without allowing waiting
    workgroups to occupy every CU and prevent the producer from being scheduled.

    The consumer addresses gates the same way the GEMM epilogue does, so this stays correct when
    the gate array belongs to a transport that spaces its gates out.
    """
    observer_id = tl.program_id(0)
    tl.debug_barrier()
    start = memrealtime()
    tl.debug_barrier()
    tl.store(observer_bounds + observer_id * 2, start)
    remaining = 1
    while remaining > 0:
        remaining = 0
        for tile in range(observer_id, num_tiles, num_observers):
            seen = tl.load(observation_order + tile) != 0
            if not seen:
                gate = tl.load(signal_of_tile + tile)
                producers = tl.load(signal_producers + gate)
                ready = tl.atomic_add(
                    gates + gate_base + gate * gate_stride, 0, sem="acquire", scope="sys"
                ) >= producers
                if ready:
                    rank = tl.atomic_add(observation_counter, 1)
                    tl.store(observation_order + tile, rank + 1)
                    row = (tile // tiles_n) * block_m
                    col = (tile % tiles_n) * block_n
                    tl.store(
                        observed + tile,
                        tl.load(output + row * output_stride + col),
                    )
                else:
                    remaining += 1
    tl.debug_barrier()
    end = memrealtime()
    tl.debug_barrier()
    tl.store(observer_bounds + observer_id * 2 + 1, end)


def save_trigger_heatmap(
    signal_times, observer_ranks, tile_schedule, num_observers, output_path
):
    """Compare release, observation and published tile orders."""
    release_order = np.empty(tile_schedule.num_tiles, dtype=np.int64)
    release_order[np.argsort(signal_times, kind="stable")] = np.arange(
        tile_schedule.num_tiles
    )
    observer_order = observer_ranks.astype(np.int64) - 1
    scheduled_order = np.asarray(tile_schedule.wg_of_tile(), dtype=np.int64)
    heatmaps = [
        release_order.reshape(tile_schedule.tiles_m, tile_schedule.tiles_n),
        observer_order.reshape(tile_schedule.tiles_m, tile_schedule.tiles_n),
        scheduled_order.reshape(tile_schedule.tiles_m, tile_schedule.tiles_n),
    ]

    width = max(14.0, min(30.0, tile_schedule.tiles_n * 1.26))
    height = max(4.5, min(12.0, tile_schedule.tiles_m * 0.42))
    fig, axes = plt.subplots(1, 3, figsize=(width, height), constrained_layout=True)
    titles = [
        "Measured gate-release order",
        f"Consumer-observed order ({num_observers} pollers)",
        "tritonblas.schedule order",
    ]
    image = None
    midpoint = (tile_schedule.num_tiles - 1) / 2
    for ax, heatmap, title in zip(axes, heatmaps, titles):
        image = ax.imshow(
            heatmap,
            cmap="viridis",
            origin="upper",
            aspect="equal",
            vmin=0,
            vmax=tile_schedule.num_tiles - 1,
        )
        ax.set(title=title, xlabel="Output tile N", ylabel="Output tile M")
        if heatmap.size <= 256:
            for tile_m in range(tile_schedule.tiles_m):
                for tile_n in range(tile_schedule.tiles_n):
                    value = int(heatmap[tile_m, tile_n])
                    color = "black" if value > midpoint else "white"
                    ax.text(
                        tile_n,
                        tile_m,
                        str(value),
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=6,
                    )
    colorbar = fig.colorbar(image, ax=axes)
    colorbar.set_label("Order (0 = first)")
    fig.suptitle(
        "GEMM tile order: measured trigger vs published schedule\n"
        f"tile {tile_schedule.block_m}×{tile_schedule.block_n}×"
        f"{tile_schedule.block_k}; grid {tile_schedule.tiles_m}×"
        f"{tile_schedule.tiles_n} = {tile_schedule.num_tiles} tiles"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return release_order, observer_order


def make_signal_layout(
    tile_schedule,
    layout,
    signal_tiles_m,
    signal_tiles_n,
    seed=0,
):
    """Map every output tile to a signal, including deliberately irregular layouts."""
    if signal_tiles_m <= 0 or signal_tiles_n <= 0:
        raise ValueError("signal tile dimensions must be positive")
    group_size = signal_tiles_m * signal_tiles_n
    tile_ids = np.arange(tile_schedule.num_tiles, dtype=np.int32)
    tile_m = tile_ids // tile_schedule.tiles_n
    tile_n = tile_ids % tile_schedule.tiles_n

    if layout == "block":
        signal_blocks_n = triton.cdiv(tile_schedule.tiles_n, signal_tiles_n)
        signal_of_tile = (
            (tile_m // signal_tiles_m) * signal_blocks_n
            + tile_n // signal_tiles_n
        )
    elif layout == "row-major":
        signal_of_tile = tile_ids // group_size
    elif layout == "dispatch":
        signal_of_tile = np.empty(tile_schedule.num_tiles, dtype=np.int32)
        signal_of_tile[np.asarray(tile_schedule.tile_of_wg)] = (
            np.arange(tile_schedule.num_tiles, dtype=np.int32) // group_size
        )
    elif layout == "modulo":
        num_signals = triton.cdiv(tile_schedule.num_tiles, group_size)
        signal_of_tile = tile_ids % num_signals
    elif layout == "random":
        rng = np.random.default_rng(seed)
        shuffled_tiles = rng.permutation(tile_schedule.num_tiles)
        signal_of_tile = np.empty(tile_schedule.num_tiles, dtype=np.int32)
        signal_of_tile[shuffled_tiles] = (
            np.arange(tile_schedule.num_tiles, dtype=np.int32) // group_size
        )
    else:
        raise ValueError(f"unknown signal layout: {layout}")

    return np.asarray(signal_of_tile, dtype=np.int32)


def example_async_tile_trigger(
    m,
    n,
    k,
    output_path,
    block_m=None,
    block_n=None,
    block_k=None,
    num_observers=8,
    gate_stride=1,
    gate_base=0,
    gate_dtype=torch.int32,
    signal_tiles_m=1,
    signal_tiles_n=1,
    signal_layout="block",
    signal_seed=0,
):
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16).T
    output = torch.empty((m, n), device="cuda", dtype=torch.float16)

    selector = tritonblas.OrigamiMatmulSelector(
        m, n, k, a.dtype, b.dtype, output.dtype, a.device
    )
    selected_tile = (selector.block_m, selector.block_n, selector.block_k)
    requested_tile = (
        block_m if block_m is not None else selector.block_m,
        block_n if block_n is not None else selector.block_n,
        block_k if block_k is not None else selector.block_k,
    )
    if requested_tile != selected_tile:
        selector._result.config.mt.m = requested_tile[0]
        selector._result.config.mt.n = requested_tile[1]
        selector._result.config.mt.k = requested_tile[2]
        selector._select_ws_params()

    tile_schedule = tritonblas.schedule_for_selector(m, n, selector)
    signal_of_tile_cpu = make_signal_layout(
        tile_schedule,
        signal_layout,
        signal_tiles_m,
        signal_tiles_n,
        seed=signal_seed,
    )
    tile_schedule = tile_schedule.with_signal_layout(signal_of_tile_cpu)
    signal_of_tile_cpu = np.asarray(tile_schedule.signal_of_tile, dtype=np.int32)
    signal_producers_cpu = np.asarray(tile_schedule.producers, dtype=np.int32)
    num_signals = tile_schedule.num_signals
    signal_of_tile = torch.from_numpy(signal_of_tile_cpu).to("cuda")
    signal_producers = torch.from_numpy(signal_producers_cpu).to("cuda")
    # gate_stride and gate_base let the epilogue write into whatever layout a transport already
    # uses, rather than forcing the transport to hand out a dense int32 array. A 64-byte gate whose
    # ready word is the first int64 uses stride 8, with each rank based into its own region; a plain
    # counter array uses the default stride 1, base 0.
    gates = torch.zeros(
        gate_base + num_signals * gate_stride, device="cuda", dtype=gate_dtype
    )

    print(
        f"Origami selected tile {selected_tile[0]}x{selected_tile[1]}x"
        f"{selected_tile[2]}; using {tile_schedule.block_m}x"
        f"{tile_schedule.block_n}x{tile_schedule.block_k}; grid "
        f"{tile_schedule.tiles_m}x{tile_schedule.tiles_n} = "
        f"{tile_schedule.num_tiles} tiles\n"
        f"signal layout {signal_layout}, tile {signal_tiles_m}x{signal_tiles_n}: "
        f"{num_signals} signals, "
        f"{signal_producers_cpu.min()}..{signal_producers_cpu.max()} producers each\n"
        f"gate layout: {gates.numel()} x {gate_dtype} elements, stride {gate_stride}, "
        f"base {gate_base}"
    )
    config = tritonblas.matmul_preamble(selector)

    observation_order = torch.zeros(
        tile_schedule.num_tiles, device="cuda", dtype=torch.int32
    )
    observation_counter = torch.zeros(1, device="cuda", dtype=torch.int32)
    signal_times = torch.zeros(
        tile_schedule.num_tiles, device="cuda", dtype=torch.int64
    )
    observed = torch.empty(
        tile_schedule.num_tiles, device="cuda", dtype=output.dtype
    )
    observer_bounds = torch.zeros(
        num_observers * 2, device="cuda", dtype=torch.int64
    )

    # Compile both kernel variants before demonstrating the asynchronous handoff.
    tritonblas.matmul_lt(
        a,
        b,
        output,
        selector,
        config,
        gates=gates,
        signal_of_tile=signal_of_tile,
        signal_times=signal_times,
        gate_stride=gate_stride,
        gate_base=gate_base,
    )
    observe_tile_triggers[(num_observers,)](
        gates,
        signal_of_tile,
        signal_producers,
        output,
        observation_order,
        observation_counter,
        observed,
        observer_bounds,
        tile_schedule.num_tiles,
        num_observers,
        tile_schedule.tiles_n,
        tile_schedule.block_m,
        tile_schedule.block_n,
        output.stride(0),
        gate_stride,
        gate_base,
        num_warps=1,
    )
    torch.cuda.synchronize()

    # Arm the observer first. It occupies one workgroup on consumer_stream while
    # the GEMM runs independently on producer_stream. The GEMM epilogue's
    # system-scope release atomics are the only events that let it make progress.
    gates.zero_()
    observation_order.zero_()
    observation_counter.zero_()
    signal_times.zero_()
    observer_bounds.zero_()
    output.fill_(float("nan"))
    torch.cuda.synchronize()

    consumer_stream = torch.cuda.Stream()
    producer_stream = torch.cuda.Stream()
    observer_started = torch.cuda.Event(enable_timing=True)
    observer_finished = torch.cuda.Event(enable_timing=True)
    producer_started = torch.cuda.Event(enable_timing=True)
    producer_finished = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(consumer_stream):
        observer_started.record()
        observe_tile_triggers[(num_observers,)](
            gates,
            signal_of_tile,
            signal_producers,
            output,
            observation_order,
            observation_counter,
            observed,
            observer_bounds,
            tile_schedule.num_tiles,
            num_observers,
            tile_schedule.tiles_n,
            tile_schedule.block_m,
            tile_schedule.block_n,
            output.stride(0),
            gate_stride,
            gate_base,
            num_warps=1,
        )
        observer_finished.record()

    with torch.cuda.stream(producer_stream):
        producer_started.record()
        tritonblas.matmul_lt(
            a,
            b,
            output,
            selector,
            config,
            gates=gates,
            signal_of_tile=signal_of_tile,
            signal_times=signal_times,
            gate_stride=gate_stride,
            gate_base=gate_base,
        )
        producer_finished.record()
    torch.cuda.synchronize()

    # Validate the entire GEMM against a float32 reference rounded to the output
    # dtype. Report the error distribution rather than checking one convenient
    # element, and separately verify every value read by the asynchronous observer.
    reference = torch.matmul(a.float(), b.float()).to(output.dtype)
    abs_error = (output.float() - reference.float()).abs()
    rtol, atol = 1e-2, 1e-2
    allowed_error = atol + rtol * reference.float().abs()
    correct = abs_error <= allowed_error
    all_correct = bool(correct.all().item())
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()
    pass_fraction = correct.float().mean().item()
    relative_error = abs_error / reference.float().abs().clamp_min(atol)
    max_relative_error = relative_error.max().item()

    tile_ids = torch.arange(tile_schedule.num_tiles, device="cuda")
    observed_rows = (tile_ids // tile_schedule.tiles_n) * tile_schedule.block_m
    observed_cols = (tile_ids % tile_schedule.tiles_n) * tile_schedule.block_n
    expected_observed = output[observed_rows, observed_cols]
    observer_values_match = torch.equal(observed, expected_observed)

    first_tile = tile_schedule.tile_of_wg[0]
    tile_row = (first_tile // tile_schedule.tiles_n) * tile_schedule.block_m
    tile_col = (first_tile % tile_schedule.tiles_n) * tile_schedule.block_n

    bounds = observer_bounds.cpu().numpy().reshape(num_observers, 2)
    observer_ranks = observation_order.cpu().numpy()
    release_stamps = signal_times.cpu().numpy()
    if (
        np.any(observer_ranks == 0)
        or np.any(release_stamps == 0)
        or np.any(bounds[:, 0] == 0)
        or np.any(bounds[:, 1] <= bounds[:, 0])
    ):
        raise RuntimeError("did not record valid release timestamps and observer order")
    observer_origin = bounds[:, 0].min()
    release_us = (release_stamps - release_stamps.min()).astype(np.float64) / 100.0
    observer_span_us = float(bounds[:, 1].max() - observer_origin) / 100.0
    producer_start_us = observer_started.elapsed_time(producer_started) * 1000.0
    producer_end_us = observer_started.elapsed_time(producer_finished) * 1000.0
    observer_event_end_us = observer_started.elapsed_time(observer_finished) * 1000.0

    release_order, observer_order = save_trigger_heatmap(
        release_stamps, observer_ranks, tile_schedule, num_observers, output_path
    )
    log_path = output_path.with_suffix(".csv")
    inverse = tile_schedule.wg_of_tile()
    inverse_array = np.asarray(inverse, dtype=np.int64)
    predicted_ready = np.full(num_signals, -1, dtype=np.int64)
    measured_open = np.zeros(num_signals, dtype=np.int64)
    np.maximum.at(predicted_ready, signal_of_tile_cpu, inverse_array)
    np.maximum.at(measured_open, signal_of_tile_cpu, release_stamps)
    predicted_signal_order = np.empty(num_signals, dtype=np.int64)
    arming_signal_ids = np.asarray(
        [arm.signal_id for arm in tile_schedule.signal_plan.arms], dtype=np.int64
    )
    predicted_signal_order[arming_signal_ids] = np.arange(num_signals)
    if not np.array_equal(
        arming_signal_ids,
        np.argsort(predicted_ready, kind="stable"),
    ):
        raise AssertionError("published arming order disagrees with final producer positions")
    measured_signal_order = np.empty(num_signals, dtype=np.int64)
    measured_signal_order[np.argsort(measured_open, kind="stable")] = np.arange(
        num_signals
    )
    if num_signals > 1:
        signal_spearman = float(
            np.corrcoef(predicted_signal_order, measured_signal_order)[0, 1]
        )
    else:
        signal_spearman = 1.0
    signal_rank_mae = float(
        np.mean(np.abs(predicted_signal_order - measured_signal_order))
    )
    signal_log_path = output_path.with_name(output_path.stem + "_signals.csv")
    observed_values = observed.float().cpu().numpy()
    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "tile_id",
                "tile_m",
                "tile_n",
                "signal_id",
                "dispatch_position",
                "measured_release_order",
                "consumer_observed_order",
                "gate_release_us",
                "observed_value",
            ]
        )
        for tile in range(tile_schedule.num_tiles):
            writer.writerow(
                [
                    tile,
                    tile // tile_schedule.tiles_n,
                    tile % tile_schedule.tiles_n,
                    signal_of_tile_cpu[tile],
                    inverse[tile],
                    release_order[tile],
                    observer_order[tile],
                    f"{release_us[tile]:.3f}",
                    f"{observed_values[tile]:.7g}",
                ]
            )

    with signal_log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "signal_id",
                "producers",
                "predicted_ready_position",
                "predicted_ready_order",
                "measured_open_order",
                "measured_open_us",
            ]
        )
        measured_origin = measured_open.min()
        for signal in range(num_signals):
            writer.writerow(
                [
                    signal,
                    signal_producers_cpu[signal],
                    predicted_ready[signal],
                    predicted_signal_order[signal],
                    measured_signal_order[signal],
                    f"{(measured_open[signal] - measured_origin) / 100.0:.3f}",
                ]
            )

    print(
        "GPU event timeline (µs after observer-stream start event):\n"
        f"  producer started: {producer_start_us:.2f}\n"
        f"  producer finished: {producer_end_us:.2f}\n"
        f"  observer finished: {observer_event_end_us:.2f}\n"
        "Observer device-clock timeline (µs after observer kernel entry):\n"
        f"  pollers: {num_observers}\n"
        f"  measured gate-release span: {release_us.max():.2f}\n"
        f"  observer device-clock span: {observer_span_us:.2f}\n"
        "Signal readiness correlation (published schedule vs measured final producer):\n"
        f"  layout: {signal_layout}\n"
        f"  signal tile: {signal_tiles_m}x{signal_tiles_n}\n"
        f"  signals: {num_signals}\n"
        f"  Spearman rank correlation: {signal_spearman:.6f}\n"
        f"  mean absolute rank error: {signal_rank_mae:.3f}\n"
        "GEMM correctness against float32 reference rounded to fp16:\n"
        f"  elements within atol={atol:g}, rtol={rtol:g}: {pass_fraction:.6%}\n"
        f"  max absolute error: {max_abs_error:.7g}\n"
        f"  mean absolute error: {mean_abs_error:.7g}\n"
        f"  max relative error (denominator clamped at atol): "
        f"{max_relative_error:.7g}\n"
        f"  observer values match GEMM output: {observer_values_match}\n"
        f"consumer read C[{tile_row}, {tile_col}] = "
        f"{observed_values[first_tile]:.7g} after gate "
        f"{signal_of_tile_cpu[first_tile]} opened\n"
        f"wrote heatmap: {output_path}\n"
        f"wrote tile log: {log_path}\n"
        f"wrote signal log: {signal_log_path}"
    )
    if not all_correct or not observer_values_match:
        raise AssertionError("GEMM or asynchronous observer correctness check failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Asynchronously trigger a Triton consumer from a tritonBLAS GEMM tile."
    )
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument(
        "--block-m",
        type=int,
        choices=[16, 32, 64, 128, 256],
        help="Override Origami's output-tile M dimension",
    )
    parser.add_argument(
        "--block-n",
        type=int,
        choices=[16, 32, 64, 128, 256],
        help="Override Origami's output-tile N dimension",
    )
    parser.add_argument(
        "--block-k",
        type=int,
        choices=[16, 32, 64, 128, 256, 512],
        help="Override Origami's reduction-tile K dimension",
    )
    parser.add_argument(
        "--num-observers",
        type=int,
        choices=[1, 2, 4, 8, 16, 32],
        default=8,
        help="Parallel gate pollers (default: 8; bounded to avoid starving the GEMM)",
    )
    parser.add_argument(
        "--signal-tiles-m",
        type=int,
        default=1,
        help="Output tiles in M sharing one signal (default: 1)",
    )
    parser.add_argument(
        "--signal-tiles-n",
        type=int,
        default=1,
        help="Output tiles in N sharing one signal (default: 1)",
    )
    parser.add_argument(
        "--signal-layout",
        choices=["block", "row-major", "dispatch", "modulo", "random"],
        default="block",
        help=(
            "How tiles are grouped into signals. Non-block layouts use "
            "--signal-tiles-m × --signal-tiles-n as the target group size."
        ),
    )
    parser.add_argument(
        "--signal-seed",
        type=int,
        default=0,
        help="Random-layout seed (default: 0)",
    )
    parser.add_argument(
        "--gate-stride",
        type=int,
        default=1,
        help="Elements between consecutive gates (default: 1; use 8 for a 64-byte int64 gate)",
    )
    parser.add_argument(
        "--gate-base",
        type=int,
        default=0,
        help="This rank's element offset into the gate array (default: 0)",
    )
    parser.add_argument(
        "--gate-dtype",
        choices=["int32", "int64"],
        default="int32",
        help="Gate word type (default: int32)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("async_tile_trigger_heatmap.png"),
        help="Heatmap output path (default: async_tile_trigger_heatmap.png)",
    )
    args = parser.parse_args()
    example_async_tile_trigger(
        args.m,
        args.n,
        args.k,
        args.output,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        num_observers=args.num_observers,
        gate_stride=args.gate_stride,
        gate_base=args.gate_base,
        gate_dtype=getattr(torch, args.gate_dtype),
        signal_tiles_m=args.signal_tiles_m,
        signal_tiles_n=args.signal_tiles_n,
        signal_layout=args.signal_layout,
        signal_seed=args.signal_seed,
    )
