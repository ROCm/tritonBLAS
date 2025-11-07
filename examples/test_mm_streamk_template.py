# test_mm_streamk_template.py
# ----------------------------------------------------------
# StreamK GEMM Triton Template – Self-Contained Validation Harness
#
# Purpose:
#   This script provides a standalone, reproducible environment
#   to validate and debug the Stream-K GEMM Triton kernel template.
#
# What it does:
#   1) Builds Inductor-style meta parameters (simulating Origami tuning).
#   2) Renders the Jinja2 kernel (as Inductor’s TritonTemplate.render() would).
#   3) Saves the rendered kernel for inspection and debugging.
#   4) Optionally compiles and runs the kernel for numerical correctness
#      or micro-benchmark measurement (TFLOPs).
#
# Usage:
#   python test_mm_streamk_template.py --M 512 --N 512 --K 512 --run
#   python test_mm_streamk_template.py --M 1024 --N 1024 --K 1024 --run --bench
#
# Notes:
#   - Requires: torch, triton, jinja2
#   - Default runs a regular GEMM (STREAMK_TILES=0).
#   - Pass --streamk_tiles > 0 to enable Stream-K accumulation logic.
#   - Designed for development and debugging of Inductor-style Triton templates.
#
# ----------------------------------------------------------
# TODOs / Next Steps:
#   [ ] Integrate with Origami parameter tuner (meta auto-selection).
#   [ ] Add Stream-K path verification (multi-SM accumulation).
#   [ ] Add structured performance report (TFLOPs vs baseline).
#   [ ] Support bias + activation fusion.
#   [ ] Add pytest-compatible wrapper for automated regression tests.
#   [ ] Visualize kernel tiling grid / wave assignment (optional).
# ----------------------------------------------------------

import argparse
import pathlib
import importlib.util
import sys
import torch
import triton
import triton.language as tl
from jinja2 import Template

# ==========================================================
# Helper: simple ceil-div
# ==========================================================
def cdiv(a, b):
    return (a + b - 1) // b

# ==========================================================
# Meta builder — all constexpr parameters live here
# ==========================================================
def build_meta(args):
    meta = dict(
        BLOCK_SIZE_M=args.block_m,
        BLOCK_SIZE_N=args.block_n,
        BLOCK_SIZE_K=args.block_k,
        GROUP_SIZE_M=args.group_m,
        NUM_SMS=args.num_sms,
        NUM_XCDS=args.num_xcds,
        STREAMK_TILES=args.streamk_tiles,
        CHUNK_SIZE=args.chunk_size,
        BIAS=args.bias,
        EVEN_K=(args.K % args.block_k == 0) if args.even_k is None else bool(args.even_k),
        ACC_TYPE="tl.float32",
        CACHE_MODIFIER_A=args.cache_a,
        CACHE_MODIFIER_B=args.cache_b,
    )

    # auto-detect actual GPU SM count
    try:
        meta["NUM_SMS"] = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        pass

    # Add any post-tuning overrides here (similar to Inductor origami tuning)
    meta.update({
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "NUM_XCDS": 1,
        "CHUNK_SIZE": 1,
        "STREAMK_TILES": 0,  # smoke test first
        "BIAS": False,
        "EVEN_K": True,
        "ACC_TYPE": tl.float32,
        "CACHE_MODIFIER_A": ".ca",
        "CACHE_MODIFIER_B": ".ca",
    })

    return meta


# ==========================================================
# Render the Jinja2 template (simulate Inductor rendering)
# ==========================================================
def render_and_save(meta, out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- helper macros (Inductor-style replacements) ---
    def size(name, dim): return f"{name}_size{dim}"
    def stride(name, dim): return f"{name}_stride{dim}"
    def ptr(name): return name
    def dtype(name): return f"{name}.dtype"

    def def_kernel(*args):
        base_args = ", ".join(args)
        scalar_args = [
            "A_size0", "A_size1",
            "B_size0", "B_size1",
            "C_size0", "C_size1",
            "A_stride0", "A_stride1",
            "B_stride0", "B_stride1",
            "C_stride0", "C_stride1",
            "bias_ptr_stride0",
        ]
        constexpr_args = [
            "BLOCK_SIZE_M: tl.constexpr",
            "BLOCK_SIZE_N: tl.constexpr",
            "BLOCK_SIZE_K: tl.constexpr",
            "GROUP_SIZE_M: tl.constexpr",
            "NUM_SMS: tl.constexpr",
            "NUM_XCDS: tl.constexpr",
            "CHUNK_SIZE: tl.constexpr",
            "STREAMK_TILES: tl.constexpr",
            "EVEN_K: tl.constexpr",
            "BIAS: tl.constexpr",
            "ACC_TYPE: tl.constexpr",
            "CACHE_MODIFIER_A: tl.constexpr",
            "CACHE_MODIFIER_B: tl.constexpr",
        ]
        sig = ", ".join(base_args.split(", ") + scalar_args + constexpr_args)
        return f"@triton.jit\ndef mm_streamk_kernel({sig}):"

    def store_output(indices, acc, mask):
        idx_m, idx_n = indices
        return (
            f"tl.store(C + {idx_m}[:, None] * C_stride0 + {idx_n}[None, :] * C_stride1, "
            f"{acc}, mask={mask})"
        )

    context = {
        **meta,
        "def_kernel": def_kernel,
        "size": size,
        "stride": stride,
        "ptr": ptr,
        "dtype": dtype,
        "store_output": store_output,
    }

    src = Template(mm_streamk_source).render(context)
    src = "import triton\nimport triton.language as tl\n" + src

    out_path = pathlib.Path(out_dir) / "mm_streamk_rendered_kernel.py"
    out_path.write_text(src)
    print(f"[DEBUG] Rendered kernel saved to: {out_path}")
    return src, out_path

# ==========================================================
# Execute rendered kernel
# ==========================================================
def exec_and_run(src, M, N, K, meta, out_path):
    spec = importlib.util.spec_from_file_location("mm_streamk_rendered_kernel", out_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mm_streamk_rendered_kernel"] = module
    spec.loader.exec_module(module)
    kernel_fn = getattr(module, "mm_streamk_kernel")

    # allocate tensors
    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)
    C = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    bias = torch.empty(1, device="cuda", dtype=C.dtype)
    bias_s0 = 0
    P = torch.empty(1, device="cuda", dtype=torch.float32)
    locks = torch.empty(1, device="cuda", dtype=torch.int32)

    A_s0, A_s1 = A.stride(0), A.stride(1)
    B_s0, B_s1 = B.stride(0), B.stride(1)
    C_s0, C_s1 = C.stride(0), C.stride(1)

    grid = (
        cdiv(M, meta["BLOCK_SIZE_M"]) * cdiv(N, meta["BLOCK_SIZE_N"]),
        1, 1,
    )

    print(f"[INFO] Launching kernel with grid={grid}")
    kernel_fn[grid](
        A, B, C, bias, P, locks,
        # scalar args (sizes/strides)
        A.shape[0], A.shape[1],
        B.shape[0], B.shape[1],
        C.shape[0], C.shape[1],
        A_s0, A_s1,
        B_s0, B_s1,
        C_s0, C_s1,
        bias_s0,
        **meta
    )
    torch.cuda.synchronize()

    # correctness check
    with torch.no_grad():
        C_ref = (A.float() @ B.float()).half()  # fp32 reference -> cast to fp16
        max_diff = (C - C_ref).abs().max().item()
        mae = (C - C_ref).abs().mean().item()
        rel_err = ( (C.float() - C_ref.float()).abs() / (C_ref.float().abs() + 1e-6) ).mean().item()

    print(f"[DEBUG] max diff vs fp32-ref->fp16 = {max_diff:.3e}, "
        f"MAE = {mae:.3e}, mean rel err = {rel_err:.3e}")

    # optional stricter check
    print("allclose:", torch.allclose(C.float(), C_ref.float(), rtol=1e-2, atol=1e-2))


# ==========================================================
# CLI
# ==========================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=512)
    p.add_argument("--N", type=int, default=512)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--bias", action="store_true")
    p.add_argument("--even_k", type=int, default=None)

    p.add_argument("--block_m", type=int, default=128)
    p.add_argument("--block_n", type=int, default=128)
    p.add_argument("--block_k", type=int, default=32)
    p.add_argument("--group_m", type=int, default=8)
    p.add_argument("--num_sms", type=int, default=120)
    p.add_argument("--num_xcds", type=int, default=1)
    p.add_argument("--streamk_tiles", type=int, default=0)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--cache_a", type=str, default=".ca")
    p.add_argument("--cache_b", type=str, default=".cg")
    p.add_argument("--out_dir", type=str, default="./_rendered")
    p.add_argument("--run", action="store_true")

    args = p.parse_args()

    # Build meta BEFORE rendering
    meta = build_meta(args)

    src, out_path = render_and_save(meta, args.out_dir)
    if args.run:
        exec_and_run(src, args.M, args.N, args.K, meta, out_path)


# ==========================================================
# Template body (trimmed)
# ==========================================================
mm_streamk_source = r"""
{{def_kernel("A", "B", "C", "bias_ptr", "P", "locks")}}

    # --- sizes & strides ---
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        return

    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}
    stride_cm = {{stride("C", 0)}}
    stride_cn = {{stride("C", 1)}}
    stride_bias = {{stride("bias_ptr", 0)}}

    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        limit = (NUM_SMS // (NUM_XCDS * CHUNK_SIZE)) * (NUM_XCDS * CHUNK_SIZE)
        if pid > limit:
            # Outside of the contiguous chunked region, leave unchanged.
            pid = pid
        else:
            local_pid = pid // NUM_XCDS
            # Calculate chunk index and position within chunk
            chunk_idx = local_pid // CHUNK_SIZE
            pos_in_chunk = local_pid % CHUNK_SIZE

            # Calculate new PID
            xcd = pid % NUM_XCDS
            pid = chunk_idx * NUM_XCDS * CHUNK_SIZE + xcd * CHUNK_SIZE + pos_in_chunk

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    total_full_tiles = total_tiles - STREAMK_TILES

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = ACC_TYPE

    # ==========================================================
    # Regular tiles (no stream-k accumulation)
    # ==========================================================
    for tile_id in range(pid, total_full_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = {{ptr("A")}} + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = {{ptr("B")}} + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        {% if BIAS %}
        bias_ = {{ptr("bias_ptr")}} + rm * stride_bias
        bias = tl.load(bias_, mask=rm < M, other=0.0)
        {% endif %}

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        {% if not EVEN_K %}
        loop_k -= 1
        {% endif %}
        tl.assume(loop_k > 1)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier="{{CACHE_MODIFIER_A}}")
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier="{{CACHE_MODIFIER_A}}")

            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier="{{CACHE_MODIFIER_B}}")
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier="{{CACHE_MODIFIER_B}}")
            acc += tl.dot(a, b, input_precision="ieee")
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        {% if not EVEN_K %}
        k = loop_k
        rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        A_BASE = {{ptr("A")}} + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = {{ptr("B")}} + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        if stride_ak == 1:
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
        else:
            A_BASE = tl.multiple_of(A_BASE, (16, 1))

        if stride_bk == 1:
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
        else:
            B_BASE = tl.multiple_of(B_BASE, (1, 16))            
        a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0, cache_modifier="{{CACHE_MODIFIER_A}}")
        b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0, cache_modifier="{{CACHE_MODIFIER_B}}")
        acc += tl.dot(a, b, input_precision="ieee")
        {% endif %}

        {% if BIAS %}
        acc += bias[:, None]
        {% endif %}

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        {{store_output(("rm", "rn"), "acc", "mask")}}

    # ==========================================================
    # Stream-K tiles (multi-SM accumulation)
    # ==========================================================
    if STREAMK_TILES == 0:
        return

    rm1 = tl.arange(0, BLOCK_SIZE_M)
    rn1 = tl.arange(0, BLOCK_SIZE_N)
    rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
    P_ = {{ptr("P")}} + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
    tl.store(P_, 0.0, cache_modifier=".wt")
    tl.store({{ptr("locks")}} + pid, 0, cache_modifier=".wt")

    tl.assume(pid >= 0)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_streamk_iters = STREAMK_TILES * iters_per_tile
    streamk_iters_pcu = total_streamk_iters // NUM_SMS
    streamk_remainder_iters = total_streamk_iters % NUM_SMS
    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)

    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = {{ptr("A")}} + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = {{ptr("B")}} + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder
        if stride_ak == 1:
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
        else:
            A_BASE = tl.multiple_of(A_BASE, (16, 1))

        if stride_bk == 1:
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
        else:
            B_BASE = tl.multiple_of(B_BASE, (1, 16))

        {% if BIAS %}
        bias_ = {{ptr("bias_ptr")}} + rm * stride_bias
        bias = tl.load(bias_, mask=rm < M, other=0.0)
        {% endif %}

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            {% if EVEN_K %}
            a = tl.load(A_BASE, cache_modifier="{{CACHE_MODIFIER_A}}")
            b = tl.load(B_BASE, cache_modifier="{{CACHE_MODIFIER_B}}")
            {% else %}
            global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
            k_mask = global_k_offset + rk < K
            a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0, cache_modifier="{{CACHE_MODIFIER_A}}")
            b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0, cache_modifier="{{CACHE_MODIFIER_B}}")
            {% endif %}

            acc += tl.dot(a, b, input_precision="ieee")
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile

        if start_iter != tile_iter:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = {{ptr("P")}} + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc, cache_modifier=".wt")
            tl.debug_barrier()
            tl.store({{ptr("locks")}} + pid, 1, cache_modifier=".wt")
        else:
            next_pid = pid + 1
            tile_iter_end = tile_iter + iters_per_tile
            end = end_iter

            # First split in M direction
            acc_m_reshaped = tl.reshape(acc, (2, BLOCK_SIZE_M // 2, BLOCK_SIZE_N))
            acc_m_permuted = tl.permute(acc_m_reshaped, (1, 2, 0))  # (M//2, N, 2)
            acc_top, acc_bottom = tl.split(acc_m_permuted)  # Split along last dimension

            # Remove singleton dimension - each is now (M//2, N)
            acc_top = tl.reshape(acc_top, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N))
            acc_bottom = tl.reshape(acc_bottom, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N))

            # Now split each half in N direction
            acc_top_reshaped = tl.reshape(acc_top, (BLOCK_SIZE_M // 2, 2, BLOCK_SIZE_N // 2))
            acc_top_permuted = tl.permute(acc_top_reshaped, (0, 2, 1))  # (M//2, N//2, 2)
            acc00, acc01 = tl.split(acc_top_permuted)  # Split along last dimension

            acc_bottom_reshaped = tl.reshape(acc_bottom, (BLOCK_SIZE_M // 2, 2, BLOCK_SIZE_N // 2))
            acc_bottom_permuted = tl.permute(acc_bottom_reshaped, (0, 2, 1))  # (M//2, N//2, 2)
            acc10, acc11 = tl.split(acc_bottom_permuted)  # Split along last dimension

            # Remove singleton dimensions - each is now (M//2, N//2)
            acc00 = tl.reshape(acc00, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
            acc01 = tl.reshape(acc01, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
            acc10 = tl.reshape(acc10, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))
            acc11 = tl.reshape(acc11, (BLOCK_SIZE_M // 2, BLOCK_SIZE_N // 2))

            while (end < tile_iter_end and next_pid < NUM_SMS):
                while tl.load({{ptr("locks")}} + next_pid, cache_modifier=".cv", volatile=True) != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                
                # Load P in two halves
                # Then for loading P data, you'd need to load 4 quadrants:
                P_base = {{ptr("P")}} + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N
                
                # Quadrant 00 (top-left)
                P_00 = P_base + tl.arange(0, BLOCK_SIZE_M // 2)[:, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 2)[None, :]
                acc00 += tl.load(P_00, cache_modifier=".cv")

                # Quadrant 01 (top-right)
                P_01 = P_base + tl.arange(0, BLOCK_SIZE_M // 2)[:, None] * BLOCK_SIZE_N + (tl.arange(0, BLOCK_SIZE_N // 2)[None, :] + BLOCK_SIZE_N // 2)
                acc01 += tl.load(P_01, cache_modifier=".cv")

                # Quadrant 10 (bottom-left)
                P_10 = P_base + (tl.arange(0, BLOCK_SIZE_M // 2)[:, None] + BLOCK_SIZE_M // 2) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 2)[None, :]
                acc10 += tl.load(P_10, cache_modifier=".cv")

                # Quadrant 11 (bottom-right)
                P_11 = P_base + (tl.arange(0, BLOCK_SIZE_M // 2)[:, None] + BLOCK_SIZE_M // 2) * BLOCK_SIZE_N + (tl.arange(0, BLOCK_SIZE_N // 2)[None, :] + BLOCK_SIZE_N // 2)
                acc11 += tl.load(P_11, cache_modifier=".cv")

                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            {% if BIAS %}
            # Split bias for top and bottom halves
            bias_top = bias[:BLOCK_SIZE_M // 2]
            bias_bottom = bias[BLOCK_SIZE_M // 2:]

            bias_top_reshaped = tl.reshape(bias_top, (BLOCK_SIZE_M // 2, 1))
            bias_bottom_reshaped = tl.reshape(bias_bottom, (BLOCK_SIZE_M // 2, 1))

            acc00 += bias_top_reshaped
            acc01 += bias_top_reshaped
            acc10 += bias_bottom_reshaped
            acc11 += bias_bottom_reshaped
            {% endif %}

            # Store all 4 quadrants
            rm_top = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M // 2)) % M
            rm_bottom = (pid_m * BLOCK_SIZE_M + tl.arange(BLOCK_SIZE_M // 2, BLOCK_SIZE_M)) % M
            rn_left = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // 2)) % N
            rn_right = (pid_n * BLOCK_SIZE_N + tl.arange(BLOCK_SIZE_N // 2, BLOCK_SIZE_N)) % N

            # Store quadrant 00 (top-left)
            mask00 = (rm_top < M)[:, None] & (rn_left < N)[None, :]
            {{store_output(("rm_top", "rn_left"), "acc00", "mask00")}}

            # Store quadrant 01 (top-right)
            mask01 = (rm_top < M)[:, None] & (rn_right < N)[None, :]
            {{store_output(("rm_top", "rn_right"), "acc01", "mask01")}}

            # Store quadrant 10 (bottom-left)
            mask10 = (rm_bottom < M)[:, None] & (rn_left < N)[None, :]
            {{store_output(("rm_bottom", "rn_left"), "acc10", "mask10")}}

            # Store quadrant 11 (bottom-right)
            mask11 = (rm_bottom < M)[:, None] & (rn_right < N)[None, :]
            {{store_output(("rm_bottom", "rn_right"), "acc11", "mask11")}}


        start_iter = end_iter
"""

# ==========================================================
if __name__ == "__main__":
    main()