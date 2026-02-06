import torch
import triton
import tritonblas
import argparse


def example_fused_matmul_lt(m, n, k, p):
    """
    Example using the peak performance API with explicit selector.
    
    Computes: D = (A @ B) @ E
    Where:
        A: M × K
        B: K × N
        E: N × P
        D: M × P (output)
    """
    print(f"\n=== Peak Performance API (fused_matmul_lt) ===")
    print(f"Computing D = (A @ B) @ E")
    print(f"Dimensions: A({m}×{k}) @ B({k}×{n}) @ E({n}×{p}) = D({m}×{p})")
    
    # Allocate input tensors
    A = torch.randn(m, k, device="cuda", dtype=torch.float16)
    B = torch.randn(k, n, device="cuda", dtype=torch.float16)
    E = torch.randn(n, p, device="cuda", dtype=torch.float16)
    D = torch.zeros((m, p), device="cuda", dtype=torch.float16)
    
    # Create selector for first GEMM (M, N, K)
    selector = tritonblas.OrigamiMatmulSelector(
        m, n, k, A.dtype, B.dtype, D.dtype, A.device
    )
    
    print(f"Selected configuration: {selector.block_m}×{selector.block_n}×{selector.block_k}")
    
    # Run fused matmul
    tritonblas.fused_matmul_lt(A, B, E, D, selector)
    
    # Verify correctness against PyTorch
    C_ref = torch.matmul(A, B)
    D_ref = torch.matmul(C_ref, E)
    
    max_diff = torch.max(torch.abs(D - D_ref)).item()
    print(f"Max difference vs PyTorch: {max_diff:.6e}")
    
    if max_diff < 1e-2:  # Tolerance for FP16
        print("✓ Results match PyTorch!")
    else:
        print("✗ Results differ from PyTorch")
    
    return D


def example_fused_matmul(m, n, k, p):
    """
    Example using the drop-in replacement API with automatic selector creation.
    
    Computes: D = (A @ B) @ E
    """
    print(f"\n=== Drop-in Replacement API (fused_matmul) ===")
    print(f"Computing D = (A @ B) @ E")
    print(f"Dimensions: A({m}×{k}) @ B({k}×{n}) @ E({n}×{p}) = D({m}×{p})")
    
    # Allocate input tensors
    A = torch.randn(m, k, device="cuda", dtype=torch.float16)
    B = torch.randn(k, n, device="cuda", dtype=torch.float16)
    E = torch.randn(n, p, device="cuda", dtype=torch.float16)
    D = torch.zeros((m, p), device="cuda", dtype=torch.float16)
    
    # Run fused matmul (selector created automatically)
    tritonblas.fused_matmul(A, B, E, D)
    
    # Verify correctness against PyTorch
    C_ref = torch.matmul(A, B)
    D_ref = torch.matmul(C_ref, E)
    
    max_diff = torch.max(torch.abs(D - D_ref)).item()
    print(f"Max difference vs PyTorch: {max_diff:.6e}")
    
    if max_diff < 1e-2:  # Tolerance for FP16
        print("✓ Results match PyTorch!")
    else:
        print("✗ Results differ from PyTorch")
    
    return D


def compare_with_sequential(m, n, k, p):
    """
    Compare fused operation with sequential matmuls.
    """
    print(f"\n=== Comparison: Fused vs Sequential ===")
    
    # Allocate tensors
    A = torch.randn(m, k, device="cuda", dtype=torch.float16)
    B = torch.randn(k, n, device="cuda", dtype=torch.float16)
    E = torch.randn(n, p, device="cuda", dtype=torch.float16)
    
    # Sequential approach
    C_seq = torch.zeros((m, n), device="cuda", dtype=torch.float16)
    D_seq = torch.zeros((m, p), device="cuda", dtype=torch.float16)
    
    selector1 = tritonblas.OrigamiMatmulSelector(m, n, k, A.dtype, B.dtype, C_seq.dtype, A.device)
    selector2 = tritonblas.OrigamiMatmulSelector(m, p, n, C_seq.dtype, E.dtype, D_seq.dtype, A.device)
    
    tritonblas.matmul_lt(A, B, C_seq, selector1)
    tritonblas.matmul_lt(C_seq, E, D_seq, selector2)
    
    # Fused approach
    D_fused = torch.zeros((m, p), device="cuda", dtype=torch.float16)
    tritonblas.fused_matmul_lt(A, B, E, D_fused, selector1)
    
    # Compare results
    max_diff = torch.max(torch.abs(D_seq - D_fused)).item()
    print(f"Max difference (Sequential vs Fused): {max_diff:.6e}")
    
    if max_diff < 1e-5:
        print("✓ Sequential and Fused produce identical results!")
    else:
        print("✗ Results differ between Sequential and Fused")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example TritonBLAS fused matrix multiplication: D = (A @ B) @ E"
    )
    parser.add_argument(
        "--m",
        type=int,
        default=2048,
        help="Number of rows in matrix A and D (default: 2048)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2048,
        help="Shared dimension: columns of B and rows of E (default: 2048)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2048,
        help="Number of columns in matrix A and rows of B (default: 2048)",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=2048,
        help="Number of columns in matrix E and D (default: 2048)",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TritonBLAS Fused Matrix Multiplication Example")
    print("=" * 70)
    
    # Run examples
    example_fused_matmul_lt(args.m, args.n, args.k, args.p)
    example_fused_matmul(args.m, args.n, args.k, args.p)
    compare_with_sequential(args.m, args.n, args.k, args.p)
    
    print("\n" + "=" * 70)
    print("Note: The kernel implementation is a stub and needs to be completed.")
    print("Once implemented, this will perform the fused operation on the GPU.")
    print("=" * 70)
