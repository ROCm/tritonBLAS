"""
Minimal example demonstrating custom epilogue functions in tritonBLAS.

This example shows how to:
1. Define your own custom epilogue function
2. Pass it to the persistent GEMM kernel
3. Verify the results
"""
import torch
import triton
import triton.language as tl
from tritonblas.kernels.persistent_gemm import persistent_matmul


# ============================================================================
# Define Custom Epilogue Function
# ============================================================================

@triton.jit
def my_custom_clamp(acc):
    """
    Custom epilogue: clamp values between -1 and 1.
    
    This is a simple example showing how to create your own epilogue function.
    You can perform any element-wise operation on the accumulator.
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
    
    Returns:
        Clamped accumulator
    """
    return tl.minimum(tl.maximum(acc, -1.0), 1.0)


# ============================================================================
# Helper Function to Run GEMM with Custom Epilogue
# ============================================================================

def matmul_with_custom_epilogue(A, B, epilogue_fn=None):
    """
    Perform matrix multiplication with a custom epilogue function.
    
    Args:
        A: Input matrix A [M, K]
        B: Input matrix B [K, N] (transposed)
        epilogue_fn: Custom epilogue function to apply
    
    Returns:
        Output matrix C [M, N]
    """
    M, K = A.shape
    _, N = B.shape
    C = torch.zeros((M, N), device="cuda", dtype=A.dtype)
    
    # Get device properties
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    
    # Fixed block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Define grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # Launch kernel with custom epilogue
    persistent_matmul[grid](
        A, B, C,
        None, None,  # No quantization scales
        A,           # Dummy bias pointer (not used)
        M, N, K,
        A.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        0,  # stride_bias (not used)
        A.stride(1), B.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SMS=num_sms,
        NUM_XCDS=1,
        CHUNK_SIZE=1,
        BIAS=False,
        EVEN_K=(K % BLOCK_SIZE_K == 0),
        CACHE_MODIFIER_A=".cg",
        CACHE_MODIFIER_B=".cg",
        epilogue_fn=epilogue_fn,  # Pass your custom epilogue here!
        QUANTIZED=False,
    )
    
    return C


# ============================================================================
# Main Example
# ============================================================================

def main():
    print("\n" + "="*70)
    print("Custom Epilogue Function Example")
    print("="*70 + "\n")
    
    # Problem size
    M, N, K = 512, 512, 512
    
    # Allocate input matrices
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(N, K, device="cuda", dtype=torch.float16).T
    
    # ========================================================================
    # Example: Custom Clamp Epilogue
    # ========================================================================
    print("Custom Clamp Epilogue (values between -1 and 1)")
    print("-" * 70)
    
    C_custom = matmul_with_custom_epilogue(A, B, epilogue_fn=my_custom_clamp)
    
    # Verify against PyTorch
    C_torch = torch.clamp(torch.matmul(A, B), -1.0, 1.0)
    max_diff = torch.max(torch.abs(C_custom - C_torch)).item()
    
    print(f"Max difference from PyTorch: {max_diff:.6f}")
    print(f"Min value: {C_custom.min().item():.6f}")
    print(f"Max value: {C_custom.max().item():.6f}")
    print(f"Sample output (first 3x3):\n{C_custom[:3, :3]}\n")



if __name__ == "__main__":
    main()
