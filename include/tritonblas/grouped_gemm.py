import torch
import triton
import triton.language as tl
import math
from .internal.wcc_grouped_gemm import wcc_groupgemm
from .origami import GroupedGemmSelector

current_device_index = torch.cuda.current_device()
current_device = torch.cuda.get_device_properties(current_device_index)
MAX_SMS = current_device.multi_processor_count

# Map torch dtypes to triton dtypes for kernel dispatch
_torch_to_triton_dtype = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def grouped_gemm(
    group_a: list[torch.Tensor],
    group_b: list[torch.Tensor],
    group_c: list[torch.Tensor] = None,
    BLK_M: int = None,
    BLK_N: int = None,
    BLK_K: int = None,
):
    """Grouped GEMM using work-centric stream-K dispatch.

    Args:
        group_a: List of A matrices (one per group).
        group_b: List of B matrices (one per group).
        group_c: Optional list of output matrices. Allocated if not provided.
        BLK_M, BLK_N, BLK_K: Optional tile sizes. If None, uses origami prediction.

    Returns:
        List of output C matrices.
    """
    group_size = len(group_a)
    assert group_size == len(group_b), "group_a and group_b must have same length"

    # Validate and collect shapes
    a_addrs, b_addrs, c_addrs = [], [], []
    g_sizes, g_lds = [], []
    in_dtype = group_a[0].dtype
    out_dtype = group_c[0].dtype if group_c is not None else in_dtype

    # Allocate output tensors if not provided
    if group_c is None:
        group_c = []
        for i in range(group_size):
            m = group_a[i].shape[0]
            n = group_b[i].shape[1]
            group_c.append(torch.empty((m, n), device="cuda", dtype=out_dtype))
    else:
        assert group_size == len(group_c), "group_c must match group_a length"

    # Collect group shapes for origami prediction
    group_shapes = []
    for i in range(group_size):
        A, B, C = group_a[i], group_b[i], group_c[i]
        assert A.shape[1] == B.shape[0], f"Group {i}: incompatible dimensions A={A.shape}, B={B.shape}"
        m, k = A.shape
        _, n = B.shape
        group_shapes.append((m, n, k))
        a_addrs.append(A.data_ptr())
        b_addrs.append(B.data_ptr())
        c_addrs.append(C.data_ptr())
        g_sizes.extend([m, n, k])
        g_lds.extend([A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1)])

    # Use origami prediction if tile sizes not specified
    if BLK_M is None or BLK_N is None or BLK_K is None:
        selector = GroupedGemmSelector(
            group_shapes, in_dtype, in_dtype, out_dtype,
            device_index=current_device_index,
        )
        BLK_M, BLK_N, BLK_K = selector.get_config()

    # Resolve triton dtype
    triton_dtype = _torch_to_triton_dtype.get(in_dtype)
    if triton_dtype is None:
        raise ValueError(f"Unsupported dtype for grouped GEMM: {in_dtype}")

    d_a_ptrs = torch.tensor(a_addrs, device="cuda", dtype=torch.int64)
    d_b_ptrs = torch.tensor(b_addrs, device="cuda", dtype=torch.int64)
    d_c_ptrs = torch.tensor(c_addrs, device="cuda", dtype=torch.int64)
    d_g_sizes = torch.tensor(g_sizes, device="cuda", dtype=torch.int32)
    d_g_lds = torch.tensor(g_lds, device="cuda", dtype=torch.int32)

    grids = MAX_SMS
    locks = torch.zeros((MAX_SMS,), device="cuda", dtype=torch.int32)
    P = torch.zeros((MAX_SMS, BLK_M * BLK_N), device="cuda", dtype=torch.float32)

    # Compute total tiles across all groups (using k-tiles for stream-K work splitting)
    total = 0
    gemm_offsets = [0]
    for m, n, k in group_shapes:
        mm = math.ceil(m / BLK_M)
        nn = math.ceil(n / BLK_N)
        kk = math.ceil(k / BLK_K)
        total += nn * mm * kk
        gemm_offsets.append(total)

    streamk_tiles_pcu = total // MAX_SMS
    streamk_remainder_tiles = total % MAX_SMS
    d_gemm_offsets = torch.tensor(gemm_offsets, dtype=torch.int32, device="cuda")

    wcc_groupgemm[(grids,)](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_gemm_offsets,
        d_g_lds,
        group_size,
        P,
        locks,
        streamk_tiles_pcu=streamk_tiles_pcu,
        streamk_remainder_tiles=streamk_remainder_tiles,
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=1,
        NUM_PRGMS=MAX_SMS,
        NUM_XCDS=8,
        MATMUL_DTYPE=triton_dtype,
    )
    return group_c
