"""
Validate AMD LDS allocation model against Triton compiled kernel metadata.

Hypothesis from TTGIR analysis:
- Triton AMD uses swizzled_shared / amd_rotating_shared (NOT PaddedSharedEncoding)
- These encodings rearrange bank mapping but add ZERO padding bytes
- For ns >= 2: pipeline allocates (ns-1) buffer sets
  LDS = (ns - 1) * (bm*bk + bk*bn) * bytes_per_elem
- For ns = 1: no pipelining, peak LDS = max(A_bytes, B_bytes)
  (A and B don't coexist — sequential inline alloc/dealloc)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bf16(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def matmul_f32(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def get_lds(kernel, bm, bn, bk, ns, dtype):
    M, N, K = 4096, 4096, 4096
    a = torch.randn(M, K, dtype=dtype, device='cuda')
    b = torch.randn(K, N, dtype=dtype, device='cuda')
    out_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    c = torch.empty(M, N, dtype=out_dtype, device='cuda')
    pgm = kernel[(1,)](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, NUM_STAGES=ns,
    )
    torch.cuda.synchronize()
    return pgm.metadata.shared


def model_lds(bm, bn, bk, bpe, ns):
    """AMD LDS model: no padding, (ns-1) pipeline buffers for ns>=2."""
    a_bytes = bm * bk * bpe
    b_bytes = bk * bn * bpe
    if ns == 1:
        return max(a_bytes, b_bytes)
    else:
        return (ns - 1) * (a_bytes + b_bytes)


if __name__ == "__main__":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.gcnArchName})")
    print(f"LDS/block: {props.shared_memory_per_block} bytes")
    print(f"Triton: {triton.__version__}")
    print()

    # bf16 configs: square, rectangular, various num_stages
    bf16_configs = [
        # Square tiles
        (64, 64, 32, 1), (64, 64, 32, 2), (64, 64, 32, 3),
        (128, 128, 64, 1), (128, 128, 64, 2), (128, 128, 64, 3),
        (256, 256, 64, 1), (256, 256, 64, 2),
        # Rectangular tiles (M != N)
        (128, 64, 64, 1), (128, 64, 64, 2),
        (64, 128, 64, 1), (64, 128, 64, 2),
        (256, 128, 64, 1), (256, 128, 64, 2),
        (128, 256, 64, 1), (128, 256, 64, 2),
        # Various BLOCK_K
        (128, 128, 32, 1), (128, 128, 32, 2),
        (128, 128, 128, 1), (128, 128, 128, 2),
        (64, 64, 64, 1), (64, 64, 64, 2),
        (64, 64, 128, 1), (64, 64, 128, 2),
    ]

    print("=" * 85)
    print(f"{'Config':>22s}  {'bpe':>3s}  {'Model':>8s}  {'Actual':>8s}  {'Match':>5s}  {'Note':>20s}")
    print("=" * 85)

    mismatches = 0
    total = 0

    print("\n--- bf16 (2 bytes/elem) ---")
    for bm, bn, bk, ns in bf16_configs:
        bpe = 2
        model = model_lds(bm, bn, bk, bpe, ns)
        try:
            actual = get_lds(matmul_bf16, bm, bn, bk, ns, torch.bfloat16)
        except Exception as e:
            actual = f"ERR"
            print(f"  {bm:>3d}x{bn:>3d}x{bk:>3d} ns={ns}   {bpe}  {model:>8d}  {'ERR':>8s}  {'':>5s}  {str(e)[:20]}")
            continue

        match = model == actual
        total += 1
        if not match:
            mismatches += 1
        note = "" if match else f"off by {actual - model:+d} ({actual/model:.3f}x)"
        sym = "OK" if match else "MISS"
        print(f"  {bm:>3d}x{bn:>3d}x{bk:>3d} ns={ns}   {bpe}  {model:>8d}  {actual:>8d}  {sym:>5s}  {note:>20s}")

    # f32 configs
    f32_configs = [
        (64, 64, 32, 1), (64, 64, 32, 2),
        (128, 128, 32, 1), (128, 128, 32, 2),
        (128, 128, 64, 1), (128, 128, 64, 2),
        (128, 64, 32, 1), (128, 64, 32, 2),
        (64, 128, 32, 1), (64, 128, 32, 2),
        (256, 256, 64, 1),
    ]

    print("\n--- f32 (4 bytes/elem) ---")
    for bm, bn, bk, ns in f32_configs:
        bpe = 4
        model = model_lds(bm, bn, bk, bpe, ns)
        try:
            actual = get_lds(matmul_f32, bm, bn, bk, ns, torch.float32)
        except Exception as e:
            print(f"  {bm:>3d}x{bn:>3d}x{bk:>3d} ns={ns}   {bpe}  {model:>8d}  {'ERR':>8s}  {'':>5s}  {str(e)[:40]}")
            continue

        match = model == actual
        total += 1
        if not match:
            mismatches += 1
        note = "" if match else f"off by {actual - model:+d} ({actual/model:.3f}x)"
        sym = "OK" if match else "MISS"
        print(f"  {bm:>3d}x{bn:>3d}x{bk:>3d} ns={ns}   {bpe}  {model:>8d}  {actual:>8d}  {sym:>5s}  {note:>20s}")

    print()
    print(f"Results: {total - mismatches}/{total} matched ({mismatches} mismatches)")
