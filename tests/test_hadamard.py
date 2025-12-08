import math
import torch
import torch.nn.functional as F
import numpy as np
from hadamard import hadamard_blocked_transform, generate_hadamard_matrix
import tritonblas 

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
def hadamard_matrix(n: int) -> np.ndarray:
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

    return H


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
    H = generate_hadamard_matrix(N).cuda() / math.sqrt(N)

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
    
    test_sizes = [32] #[64, 128, 256, 512, 1024]
    batch_sizes = [1, 4]
    
    print(f"\nRunning tests on device: {device}")
    print("=" * 60)
    
    for N in test_sizes:
        for batch in batch_sizes:
            print(f"\nTesting N={N}, batch={batch}")
            print("-" * 40)
            
            # Create random input
            x = torch.randn(batch, N, device=device, dtype=torch.float32)
            
            # 1. Matrix multiplication version
            matmul_result = fwht_matmul(x.clone()) 
            
            # 2. Python FWHT reference
            ref_result = fwht_torch_reference(x.clone(), N) 

            # 3. triton FWHT reference
            triton_result = full_hadamard_triton(x.clone()) / math.sqrt(N)

            # 4. blocked GEMM
            H = generate_hadamard_matrix(N).cuda() / math.sqrt(N)
            blocked_result = hadamard_blocked_transform(x.clone(), H) 

            # 5. fast blocked GEMM
            H = generate_hadamard_matrix(N).cuda() / math.sqrt(N)
            fast_blocked_result = tritonblas.hadamard_blocked_fast(x.clone()) 

            # matmul vs ref
            print(f"matmul vs ref: {F.mse_loss(matmul_result, ref_result).item():.2e}")

            # matmul vs triton
            print(f"matmul vs triton: {F.mse_loss(matmul_result, triton_result).item():.2e}")

            # ref vs triton
            print(f"ref vs triton: {F.mse_loss(ref_result, triton_result).item():.2e}")

            # blocked GEMM vs triton
            print(f"blocked gemm vs. triton: {F.mse_loss(blocked_result, triton_result)}")

            # blocked GEMM vs ref
            print(f"blocked gemm vs. matmul: {F.mse_loss(blocked_result, matmul_result)}")

            # fast blocked GEMM vs blocked GEMM
            print(f"fast blocked gemm vs. blocked gemm: {F.mse_loss(fast_blocked_result, blocked_result)}")

    
    print("\n" + "=" * 60)
    print("Test completed")


if __name__ == "__main__":
    test_full_hadamard()