# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Persistent GEMM kernel using tritonblas aggregates.

This kernel uses InputView and OutputView aggregates that store matrix layout
information, enabling clean separation between pointer computation and GEMM logic.
"""

import triton
import triton.language as tl
import torch
from triton.language.extra.hip import memrealtime

from tritonblas.kernels.stages import (
    ScheduleContext,
    make_schedule_context,
    GemmContext, 
    make_input_view,
    make_output_view,
    make_scale_view,
    make_bias_view,
    make_signal_view,
    make_wait_view,
)


@triton.jit()
def persistent_matmul(
    A,
    B,
    C,
    A_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    B_scale_ptr,  # Optional: None for fp16/bf16, pointer for int8/fp8
    bias_ptr,
    counter_ptr,  # Optional: int32[num_signals] for tile completion signaling
    wait_ptr,     # Optional: int32[num_waits] for producer-consumer readiness
    wait_expected_ptr,  # Optional: int32[num_waits] runtime target values
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    stride_ak,
    stride_bk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    ENABLE_COUNTER: tl.constexpr = False,
    COUNTER_NUM: tl.constexpr = 0,
    COUNTER_MAP_TYPE: tl.constexpr = 0,  # 0=identity, 1=row, 2=col, 3=block, 4=launch_wave
    COUNTER_BLOCK_GROUP_M: tl.constexpr = 1,
    COUNTER_BLOCK_GROUP_N: tl.constexpr = 1,
    ENABLE_WAIT: tl.constexpr = False,
    WAIT_NUM: tl.constexpr = 0,
    WAIT_MAP_TYPE: tl.constexpr = 0,  # 0=identity, 1=row, 2=col, 3=block
    WAIT_BLOCK_GROUP_M: tl.constexpr = 1,
    WAIT_BLOCK_GROUP_N: tl.constexpr = 1,
    WAIT_EXPECTED_INC: tl.constexpr = 1,
    QUANTIZED: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = True,
    # ─── Tile signalling (optional) ──────────────────────────────────────────
    # Advertise each output tile as it is finished, so a dependent consumer -- a
    # triggered collective with pre-armed chains, typically -- can start moving
    # it without waiting for the whole GEMM. Off by default and compiled out
    # entirely when SIGNAL is False, so an ordinary matmul is unchanged.
    gates=None,
    signal_of_tile=None,
    signal_times=None,
    SIGNAL: tl.constexpr = False,
    TRACE_SIGNAL: tl.constexpr = False,
    # Gate array geometry, so the epilogue is not tied to one transport's layout. Supported layouts
    # include plain int32 counters at unit stride and 64-byte gate structs whose ready word is the
    # first int64, i.e. a stride of 8 int64 elements. A strided torch view cannot express this because
    # Triton indexes the raw pointer and ignores torch strides, so the stride has to be a kernel
    # parameter. GATE_BASE is this rank's offset into a gate array that holds every rank's signals end
    # to end -- without it a multi-rank caller signals its neighbour's gates and its own never fill,
    # which presents as a hang in the collective rather than a wrong answer.
    GATE_STRIDE: tl.constexpr = 1,
    GATE_BASE: tl.constexpr = 0,
    SIGNAL_WRITE_THROUGH: tl.constexpr = True,
):
    """
    Persistent GEMM kernel using GemmContext aggregate.
    
    Matrix views are created using factory functions that handle any memory layout.
    Just describe your matrices and their strides - no layout flags needed.
    
    STRIDE PARAMETERS:
    ------------------
    - stride_am: A's stride when moving one M row down
    - stride_ak: A's stride in K dimension
    - stride_bk: B's stride in K dimension
    - stride_bn: B's stride when moving one N column right
    - stride_cm: C's stride when moving one M row down
    - stride_cn: C's stride in N dimension
    """

    # Determine accumulator dtype based on output type
    acc_dtype = tl.int32 if C.type.element_ty == tl.int8 else tl.float32
    
    # ════════════════════════════════════════════════════════════════════════
    # CREATE MATRIX VIEWS
    # ════════════════════════════════════════════════════════════════════════
    # Factory functions handle stride type coercion automatically
    tensorA = make_input_view(A, M, K, stride_am, stride_ak)
    tensorB = make_input_view(B, K, N, stride_bk, stride_bn)
    tensorC = make_output_view(C, M, N, stride_cm, stride_cn)
    
    # ════════════════════════════════════════════════════════════════════════
    # CREATE SYNCHRONIZATION/EPILOGUE VIEWS (optional wait, scale, bias, signal)
    # ════════════════════════════════════════════════════════════════════════
    scale_view = make_scale_view(A_scale_ptr, B_scale_ptr, M, N) if A_scale_ptr is not None else None
    bias_view = make_bias_view(bias_ptr, N, stride_bias) if BIAS else None
    wait_view = make_wait_view(wait_ptr, wait_expected_ptr) if ENABLE_WAIT else None

    # Create signal view if enabled
    signal_view = None
    if ENABLE_COUNTER:
        signal_view = make_signal_view(counter_ptr)
    
    # ════════════════════════════════════════════════════════════════════════
    # CONSTRUCT GEMM CONTEXT TO MANAGE MATH RELEVANT CONTEXT
    # ════════════════════════════════════════════════════════════════════════
    ctx = GemmContext(
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        NUM_SMS, NUM_XCDS,
        GROUP_SIZE_M, CHUNK_SIZE,
        CACHE_MODIFIER_A, CACHE_MODIFIER_B,
        acc_dtype, ALLOW_TF32, EVEN_K, QUANTIZED,
    )
    
    # ════════════════════════════════════════════════════════════════════════
    # CREATE SCHEDULE CONTEXT FROM GEMM CONTEXT TO MANAGE OUTER LOOP ITERATION
    # ════════════════════════════════════════════════════════════════════════
    sched = make_schedule_context(M, N, K, ctx)
    
    # ════════════════════════════════════════════════════════════════════════
    # PERSISTENT LOOP: Process multiple tiles per workgroup
    # ════════════════════════════════════════════════════════════════════════
    start_tile, total_tiles, stride = sched.persistent_tile_range()
    launch_program_id = tl.program_id(0)
    for tile_id in range(start_tile, total_tiles, stride):
        # ════════════════════════════════════════════════════
        # Get schedule aware output tile to be processed this loop iteration
        # ════════════════════════════════════════════════════
        out_tile = sched.get_tile_from_idx(tile_id)

        # Wait until the producer has made the batch containing this tile
        # visible in memory before entering the GEMM K-loop.
        if wait_view is not None:
            wait_view.wait_for_tile(
                out_tile, M, N,
                num_flags=WAIT_NUM,
                map_type=WAIT_MAP_TYPE,
                block_group_m=WAIT_BLOCK_GROUP_M,
                block_group_n=WAIT_BLOCK_GROUP_N,
                expected_inc=WAIT_EXPECTED_INC,
            )
        
        # ════════════════════════════════════════════════════════════════════
        # COMPUTE GEMM: K-loop handled by GemmContext
        # ════════════════════════════════════════════════════════════════════
        acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
        
        # ════════════════════════════════════════════════════════════════════
        # STORE RESULT: Epilogue (scale, bias, convert, signal) handled by OutputView
        # Store Accumulator to output matrix C at pointers defined by out_tile
        # ════════════════════════════════════════════════════════════════════
        tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view, signal=signal_view,
                     signal_num=COUNTER_NUM, signal_map_type=COUNTER_MAP_TYPE,
                     signal_block_m=COUNTER_BLOCK_GROUP_M, signal_block_n=COUNTER_BLOCK_GROUP_N,
                     signal_launch_wave_id=launch_program_id)

        # ════════════════════════════════════════════════════════════════════
        # SIGNAL: advertise this tile to a waiting consumer
        # ════════════════════════════════════════════════════════════════════
        # Three lines, each load-bearing:
        #
        # * The store above must have reached a coherence point the consumer can
        #   see. A copy engine sits on the IOD, past this XCD's L2, so a tile
        #   left dirty in L2 is invisible to it and an open gate lets it copy
        #   stale bytes -- a fast wrong answer rather than a hang. Callers that
        #   signal to an off-device consumer must pass a write-through output
        #   view; SIGNAL_WRITE_THROUGH records that requirement.
        # * `tl.debug_barrier()` so every lane's stores have retired before any
        #   lane advertises the tile.
        # * `sem="release", scope="sys"`: device scope is not enough when the
        #   consumer is not on the device.
        #
        # The gate id comes from a table indexed by linear tile id rather than
        # from a closed-form map. A closed form needs a new kernel branch per
        # grouping and the set of useful groupings is open; a table supports any
        # of them with no kernel change, and the load is issued once per tile
        # against an L2-resident array.
        #
        # `atomic_add` of 1, not a store: a tile may be one of several producers
        # of a coarser gate, and the consumer's poll threshold is the join. The
        # producers then need no barrier among themselves and may finish in any
        # order, which is what lets the GEMM keep its own tile schedule.
        if SIGNAL:
            tl.debug_barrier()
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            linear_tile = out_tile.pid_m * num_pid_n + out_tile.pid_n
            gate = tl.load(signal_of_tile + linear_tile)
            tl.atomic_add(gates + GATE_BASE + gate * GATE_STRIDE, 1,
                          sem="release", scope="sys")
            if TRACE_SIGNAL:
                tl.debug_barrier()
                stamp = memrealtime()
                tl.debug_barrier()
                tl.store(signal_times + linear_tile, stamp)
