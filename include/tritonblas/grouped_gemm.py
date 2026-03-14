import torch
import triton
import triton.language as tl
import math
from .internal.grouped_persistent_matmul import grouped_persistent_matmul
from .origami import GroupedGemmSelector, MatmulHeuristicResult
from .matmul import persistent_matmul_lt

current_device_index = torch.cuda.current_device()
current_device = torch.cuda.get_device_properties(current_device_index)
MAX_SMS = current_device.multi_processor_count

_torch_to_triton_dtype = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def _is_homogeneous(group_shapes):
    """Check if all groups have the same (M, N, K)."""
    return all(s == group_shapes[0] for s in group_shapes)


def _homogeneous_dispatch(group_a, group_b, group_c, m, n, k, group_size):
    """Fast path: all groups same shape → reuse persistent_matmul per group.

    Creates the origami selector ONCE and reuses it for all groups.
    Uses the highly optimized persistent_matmul with constexpr strides,
    achieving near-peak TFLOPS.
    """
    selector = MatmulHeuristicResult(
        m, n, k, group_a[0].dtype, group_b[0].dtype, group_c[0].dtype,
        streamk=False,
    )
    for i in range(group_size):
        persistent_matmul_lt(group_a[i], group_b[i], group_c[i], selector)


def _heterogeneous_dispatch(group_a, group_b, group_c, group_shapes, group_size,
                             BLK_M, BLK_N, BLK_K, triton_dtype):
    """Single-kernel dispatch for heterogeneous groups."""
    a_addrs = [a.data_ptr() for a in group_a]
    b_addrs = [b.data_ptr() for b in group_b]
    c_addrs = [c.data_ptr() for c in group_c]
    g_sizes, g_lds = [], []
    for i, (m, n, k) in enumerate(group_shapes):
        g_sizes.extend([m, n, k])
        g_lds.extend([group_a[i].stride(0), group_a[i].stride(1),
                       group_b[i].stride(0), group_b[i].stride(1),
                       group_c[i].stride(0), group_c[i].stride(1)])

    d_a_ptrs = torch.tensor(a_addrs, device="cuda", dtype=torch.int64)
    d_b_ptrs = torch.tensor(b_addrs, device="cuda", dtype=torch.int64)
    d_c_ptrs = torch.tensor(c_addrs, device="cuda", dtype=torch.int64)
    d_g_sizes = torch.tensor(g_sizes, device="cuda", dtype=torch.int32)
    d_g_lds = torch.tensor(g_lds, device="cuda", dtype=torch.int32)

    gemm_offsets = [0]
    for m, n, k in group_shapes:
        gemm_offsets.append(gemm_offsets[-1] + math.ceil(m / BLK_M) * math.ceil(n / BLK_N))
    d_gemm_offsets = torch.tensor(gemm_offsets, dtype=torch.int32, device="cuda")

    num_xcds = 8
    group_size_m = int(math.ceil(math.sqrt(MAX_SMS / num_xcds)))
    total_output_tiles = gemm_offsets[-1]
    chunk_size = max(1, min(group_size_m * group_size_m, total_output_tiles // num_xcds))

    grouped_persistent_matmul[(MAX_SMS,)](
        d_a_ptrs, d_b_ptrs, d_c_ptrs,
        d_g_sizes, d_gemm_offsets, d_g_lds,
        BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=group_size_m,
        GROUP_COUNT=group_size,
        NUM_SMS=MAX_SMS, NUM_XCDS=num_xcds, CHUNK_SIZE=chunk_size,
        MATMUL_DTYPE=triton_dtype,
        num_stages=2, num_warps=8,
    )


def grouped_gemm(
    group_a: list[torch.Tensor],
    group_b: list[torch.Tensor],
    group_c: list[torch.Tensor] = None,
    BLK_M: int = None,
    BLK_N: int = None,
    BLK_K: int = None,
):
    """Grouped GEMM with automatic dispatch optimization.

    - Homogeneous groups (same M,N,K): uses fast persistent_matmul per group
      with constexpr strides for near-peak TFLOPS.
    - Heterogeneous groups: single-kernel dispatch with dynamic per-group metadata.
    """
    group_size = len(group_a)
    assert group_size == len(group_b)

    in_dtype = group_a[0].dtype
    out_dtype = group_c[0].dtype if group_c is not None else in_dtype

    if group_c is None:
        group_c = []
        for i in range(group_size):
            group_c.append(torch.empty((group_a[i].shape[0], group_b[i].shape[1]),
                                       device="cuda", dtype=out_dtype))
    else:
        assert group_size == len(group_c)

    group_shapes = []
    for i in range(group_size):
        A, B = group_a[i], group_b[i]
        assert A.shape[1] == B.shape[0], f"Group {i}: incompatible A={A.shape}, B={B.shape}"
        group_shapes.append((A.shape[0], B.shape[1], A.shape[1]))

    if _is_homogeneous(group_shapes):
        m, n, k = group_shapes[0]
        _homogeneous_dispatch(group_a, group_b, group_c, m, n, k, group_size)
    else:
        if BLK_M is None or BLK_N is None or BLK_K is None:
            selector = GroupedGemmSelector(
                group_shapes, in_dtype, in_dtype, out_dtype,
                device_index=current_device_index,
            )
            BLK_M, BLK_N, BLK_K = selector.get_config()

        triton_dtype = _torch_to_triton_dtype.get(in_dtype)
        if triton_dtype is None:
            raise ValueError(f"Unsupported dtype: {in_dtype}")

        _heterogeneous_dispatch(group_a, group_b, group_c, group_shapes,
                                group_size, BLK_M, BLK_N, BLK_K, triton_dtype)

    return group_c
