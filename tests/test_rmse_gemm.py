import math
import torch
import torch.nn.functional as F
import numpy as np
from hadamard import hadamard_blocked_transform, generate_hadamard_matrix
import tritonblas 
from tritonblas.utils import quantize_tensor_per_channel
from fp4_utils import dynamic_mxfp4_quant, mxfp4_to_f32, e8m0_to_f32
import aiter 
def is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def is_triton_available() -> bool:
    """Check if Triton is available."""
    try:
        import triton
        return True
    except ImportError:
        return False


# Reference implementation
def hadamard_matrix(n: int, dtype=torch.float32) -> np.ndarray:
    """
    Create a Hadamard matrix of size n x n using Sylvester's construction.
    n must be a power of 2.
    
    Args:
        n: Size of the Hadamard matrix (must be power of 2)
    
    Returns:
        Hadamard matrix as numpy array
    """
    if n < 1:
        lg2 = 0
    else:
        lg2 = int(math.log(n, 2))
    if 2 ** lg2 != n:
        raise ValueError("n must be int and power of 2.")

    if n == 1:
        return np.array([[1]], dtype=np.float32)

    H = np.array([[1, 1], [1, -1]], dtype=np.float32)

    # Hadamard stacking via Sylvester's construction
    # H H
    # H -H
    for i in range(0, lg2 - 1):
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))

    return H.to(dtype)


def fwht_matmul(x: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication version of Hadamard transform (un-normalized).
    Power-of-two length required along last dimension.
    
    Args:
        x: Input tensor with power-of-2 size in last dimension
    
    Returns:
        Hadamard transformed tensor
    """
    *leading, N = x.shape
    
    if not is_power_of_2(N):
        raise ValueError(f"N must be power-of-two, got {N}")
    
    # Create Hadamard matrix
    H = generate_hadamard_matrix(N, dtype=x.dtype).cuda() / math.sqrt(N)

    # H_tensor = torch.from_numpy(H).to(x.device).to(x.dtype)
    # Perform matrix multiplication: x @ H
    return x @ H


def fwht_torch_reference(x: torch.Tensor, N: int = 0) -> torch.Tensor:
    """
    Reference FWHT implementation matching Sylvester's Hadamard construction (un-normalized).
    Power-of-two length required along last dimension.
    """
    y = x.clone()
    if not N:
        N = y.shape[-1]
    
    # Perform butterfly operations matching Sylvester's construction
    # Start with stride N/2 and work down to stride 1
    stride = N // 2
    while stride >= 1:
        for start in range(0, N, stride * 2):
            for i in range(stride):
                idx1 = start + i
                idx2 = start + i + stride
                a = y[..., idx1].clone()
                b = y[..., idx2].clone()
                y[..., idx1] = a + b
                y[..., idx2] = a - b
        stride //= 2
    
    y = y / math.sqrt(N)
    return y


# Triton kernels
if is_triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _fwht_tile_kernel(X_ptr, stride_row, N, TILE: tl.constexpr, LOG_TILE: tl.constexpr):
        """Blockwise FWHT (per-tile) with masked final (partial) tile."""
        pid = tl.program_id(0)
        tiles_per_row = tl.cdiv(N, TILE)
        row = pid // tiles_per_row
        tile_id = pid % tiles_per_row
        row_base = row * stride_row
        start = tile_id * TILE

        offs = tl.arange(0, TILE)
        idx_i = start + offs
        valid_i = idx_i < N

        # Iterate butterfly stages
        for s in range(LOG_TILE):
            dist = 1 << s
            idx_j = idx_i ^ dist
            valid_j = idx_j < N
            process = (offs & dist) == 0

            # Load both ends
            vi = tl.load(X_ptr + row_base + idx_i, mask=valid_i, other=0.0)
            vj = tl.load(X_ptr + row_base + idx_j, mask=valid_j, other=0.0)

            vsum = vi + vj
            vdiff = vi - vj
            # Store results
            tl.store(X_ptr + row_base + idx_i, vsum, mask=valid_i & process)
            tl.store(X_ptr + row_base + idx_j, vdiff, mask=valid_j & process)

    @triton.jit
    def _fwht_merge_kernel(X_ptr, stride_row, N, STAGE_DIST: tl.constexpr):
        """Merge stage across tiles."""
        pid = tl.program_id(0)
        groups_per_row = N // (2 * STAGE_DIST)
        row_id = pid // groups_per_row
        group_id = pid % groups_per_row
        base0 = row_id * stride_row + group_id * 2 * STAGE_DIST
        base1 = base0 + STAGE_DIST
        offs = tl.arange(0, STAGE_DIST)
        a = tl.load(X_ptr + base0 + offs)
        b = tl.load(X_ptr + base1 + offs)
        tl.store(X_ptr + base0 + offs, a + b)
        tl.store(X_ptr + base1 + offs, a - b)


def _select_block(N: int, candidate_blocks=(32, 64, 128), max_mono_kernel_n=2048, block_size=None):
    """Choose tile size for tiled FWHT."""
    divs = [b for b in candidate_blocks if b <= N and N % b == 0]
    if not divs:
        return N
    if block_size and block_size in divs:
        return block_size
    if (N in divs) and (N <= max_mono_kernel_n) and not block_size:
        return N
    small = [b for b in divs if b <= 128]
    if small:
        return sorted(small, reverse=True)[0]
    return max(divs)

def torch_rmsnorm(x, g, out_dtype=torch.float16, epsilon=1e-6):
    M, N = x.shape
    # cast to float32 as the triton kernel
    x_f32 = x.float()
    g_f32 = g.float()
    rms = torch.sqrt(torch.sum(x_f32 * x_f32, dim=-1) * 1 / N)
    rsigma = 1.0 / rms
    rms_norm_f32 = x_f32 * rsigma.unsqueeze(1) * g_f32
    rms_norm = rms_norm_f32.to(out_dtype)
    return rms_norm

def make_outlier_tensor(shape, seed=0, outlier_ratio=0.01):
    """Create a tensor with outliers in random rows for testing (row-wise outlier)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randn(shape, generator=g, device="cuda", dtype=torch.bfloat16)
    
    # Create mask based on random rows (axis 0)
    num_rows = shape[0]
    num_outlier_rows = max(1, int(num_rows * outlier_ratio))
    
    # Randomly select rows to be outliers
    outlier_indices = torch.randperm(num_rows, generator=g, device="cuda")[:num_outlier_rows]
    
    # Create mask: 1 for outlier rows, 0 for others
    mask = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    mask[outlier_indices, :] = 1.0
    
    spikes = torch.randn(shape, generator=g, device="cuda", dtype=torch.bfloat16) * 25.0
    return base + mask * spikes

def full_hadamard_triton(x: torch.Tensor, block_size=None):
    """
    Full Hadamard transform using Triton kernels.
    Un-normalized. Requires power-of-two length.
    """
    if not is_triton_available():
        raise RuntimeError("Triton not available")
    
    assert x.is_contiguous()
    *leading, N = x.shape
    
    if not is_power_of_2(N):
        raise ValueError(f"N must be power-of-two, got {N}")
    
    rows = int(torch.tensor(leading).prod()) if leading else 1
    BLOCK = _select_block(N, block_size=block_size)
    LOG_BLOCK = int(math.log2(BLOCK))
    tiles_per_row = N // BLOCK
    total_intra = rows * tiles_per_row
    stride_row = N

    # Intra-tile FWHT
    _fwht_tile_kernel[(total_intra,)](x.view(-1, N), stride_row, N, TILE=BLOCK, LOG_TILE=LOG_BLOCK)

    # If single tile (BLOCK == N), we're done
    if BLOCK == N:
        return x

    # Inter-tile merges
    dist = BLOCK
    while dist < N:
        groups_per_row = N // (2 * dist)
        total_groups = rows * groups_per_row
        _fwht_merge_kernel[(total_groups,)](x.view(-1, N), stride_row, N, STAGE_DIST=dist)
        dist *= 2
    
    return x


def test_full_hadamard():
    """Test full Hadamard: matrix multiplication, PyTorch reference, and Triton kernel."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    test_sizes = [16384] #[32] #, 64, 128, 256, 512, 1024]
    batch_sizes = [1] #, 4, 32, 64]
    
    print(f"\nRunning tests on device: {device}")
    print("=" * 60)
    
    # block_m_sizes = [64, 128, 256]
    # block_n_sizes = [64, 128, 256]
    # block_k_sizes = [128, 256, 512]
    block_m = 256
    block_n = 256
    block_k = 256
    for N in test_sizes:
        for batch in batch_sizes:
            print(f"\nTesting N={N}, batch={batch}")
            print("-" * 40)
            
            K = N
            M = N
            # Create outlier tensor
            x = make_outlier_tensor((batch,N))
            x = x.to(dtype)

            # x = torch.randn(batch, N, device=device, dtype=dtype)
            # quant_dtype = torch.float8_e4m3fnuz
            quant_dtype = torch.float8_e4m3fn
            x_quant = torch.empty(x.shape, dtype=quant_dtype, device=device)
            x_fp8 = torch.empty(x.shape, dtype=quant_dtype, device=device)
            # x_quant = torch.empty(x.shape, dtype=dtype, device=device)
            x_fp8_scales = torch.empty(x.shape[0], 1, dtype=torch.float32, device=device)
            x_quant_scales = torch.empty(x.shape[0], 1, dtype=torch.float32, device=device)
            w = torch.randn((N, K), device=device, dtype=dtype)
            w_fp8 = torch.empty(w.shape, dtype=quant_dtype, device=device)
            w_fp8_scales = torch.empty(w.shape[0], 1, dtype=torch.float32, device=device)
            # w_fp8, w_fp8_scales =  quantize_tensor_per_channel(w.clone(), quant_dtype, axis=1)
            aiter.ops.triton.quant.dynamic_per_tensor_quant_fp8_i8(w_fp8, w, w_fp8_scales)
            w_fp4, w_scales = dynamic_mxfp4_quant(w)
            out_fp16 = torch.empty((M, N), device=device, dtype=dtype)
            out_fp8 = torch.empty((M, N), device=device, dtype=dtype)
            out_fp4 = torch.empty((M, N), device=device, dtype=dtype)
            out_fp4_fused = torch.empty((M, N), device=device, dtype=dtype)
            out_had_unfused = torch.empty((M, N), device=device, dtype=dtype)
            out_had_fused = torch.empty((M, N), device=device, dtype=dtype)
            
            # rmsnorm parameters
            weight = torch.randn(N, dtype=dtype, device=device)
            eps = 1e-5
            use_model_sensitive_rmsnorm = 0

            # 1. FP16 RMSNorm + FP16 GEMM
            x_norm = tritonblas.rms_norm(x.clone(), weight, eps, use_model_sensitive_rmsnorm)
            tritonblas.matmul(x_norm, w, out_fp16)

            # 1. FP16 RMSNorm + FP8 GEMM
            x_norm = tritonblas.rms_norm(x.clone(), weight, eps, use_model_sensitive_rmsnorm)
            # x_fp82, x_fp8_scales2 =  quantize_tensor_per_channel(x_norm.clone(), quant_dtype, axis=1)
            aiter.ops.triton.quant.dynamic_per_tensor_quant_fp8_i8(x_fp8, x_norm, x_fp8_scales)
            selector = tritonblas.MatmulHeuristicResult(
                M, N, K, x_fp8.dtype, w_fp8.dtype, out_fp8.dtype
            )
            tritonblas.matmul_a8w8_lt(x_fp8, w_fp8, x_fp8_scales, w_fp8_scales, out_fp8, selector)

            # 2. FP16 RMSNorm + quant + FP4 GEMM
            x_norm = tritonblas.rms_norm(x, weight, eps, use_model_sensitive_rmsnorm)
            x_fp4_v1, x_scales_v1 = dynamic_mxfp4_quant(x_norm)
            tritonblas.matmul_fp4(
                x_fp4_v1, w_fp4, out_fp4, x_scales_v1, w_scales,
                block_m=block_m, block_n=block_n, block_k=block_k
            )

            # 3. fused (FP16 RMSNorm + quant) + FP4 GEMM
            x_norm = tritonblas.rmsnorm2d_fwd_with_dynamicquant(x_quant, x.clone(), x_quant_scales, weight, eps)
            tritonblas.matmul_fp4(
                x_fp4_v1, w_fp4, out_fp4_fused, x_scales_v1, w_scales,
                block_m=block_m, block_n=block_n, block_k=block_k
            )

            # 4. unfused rmsnorm + hadamard + quant + FP4 GEMM
            x_norm_1 = tritonblas.rms_norm(x, weight, eps, use_model_sensitive_rmsnorm)
            x_had = tritonblas.hadamard_blocked_fast(x_norm_1) 
            x_fp4_v1, x_scales_v1 = dynamic_mxfp4_quant(x_had)
            tritonblas.matmul_fp4(
                x_fp4_v1, w_fp4, out_had_unfused, x_scales_v1, w_scales,
                block_m=block_m, block_n=block_n, block_k=block_k
            )

            # 5. fused (rmsnorm + hadamard + quant) + FP4 GEMM
            (x_fp4, x_scales), _, _ = tritonblas.fused_rms_hadamard_mxfp4_quant(x.clone(), weight, eps)
            tritonblas.matmul_fp4(
                x_fp4, w_fp4, out_had_fused, x_scales, w_scales,
                block_m=block_m, block_n=block_n, block_k=block_k
            )
            # (x_fp4, x_scales), _, _ = tritonblas.fused_mxfp4_quant(x.clone(), weight, eps)

            print(f"||out||: {torch.norm(out_fp16)}")
            # print(f"FP16 vs. FP8: {F.mse_loss(x_norm, x_fp8.to(torch.float32))}")
            print(f"FP16 vs. FP8: {F.mse_loss(out_fp16, out_fp8)}")
            print(f"FP16 vs. FP4: {F.mse_loss(out_fp16, out_fp4)}")
            print(f"FP16 vs. FP4 fused: {F.mse_loss(out_fp16, out_fp4_fused)}")
            print(f"FP16 vs. FP4 hadamard: {F.mse_loss(out_fp16, out_had_unfused)}")
            print(f"FP16 vs. FP4 hadamard fused: {F.mse_loss(out_fp16, out_had_fused)}")

    
    print("\n" + "=" * 60)
    print("Test completed")


if __name__ == "__main__":
    test_full_hadamard()