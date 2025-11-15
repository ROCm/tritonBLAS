"""
Utility functions for composable Triton GEMM kernels.
"""
import triton
import triton.language as tl


@triton.jit
def pid_identity(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE):
    """Identity PID mapping - returns the PID unchanged."""
    return pid


@triton.jit
def pid_chiplet_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE):
    """
    Chiplet-aware PID mapping using chunked transform.
    
    This function redistributes work across chiplets (XCDs) in chunks to improve
    L2 cache locality. PIDs are remapped so that consecutive chunks are assigned
    to the same chiplet.
    """
    if pid > (NUM_SMS // (NUM_XCDS * CHUNK_SIZE)) * (NUM_XCDS * CHUNK_SIZE):
        # Outside of the contiguous chunked region, leave unchanged.
        return pid
    
    local_pid = pid // NUM_XCDS 
    # Calculate chunk index and position within chunk
    chunk_idx = local_pid // CHUNK_SIZE 
    pos_in_chunk = local_pid % CHUNK_SIZE 

    # Calculate new PID
    xcd = pid % NUM_XCDS 
    new_pid = chunk_idx * NUM_XCDS * CHUNK_SIZE + xcd * CHUNK_SIZE + pos_in_chunk
    return new_pid


@triton.jit
def dot_acc(acc, a, b, QUANTIZED: tl.constexpr, ALLOW_TF32: tl.constexpr):
    """
    Accumulate dot product with appropriate precision.
    
    Args:
        acc: Accumulator tensor
        a: Left operand
        b: Right operand
        QUANTIZED: If True, use IEEE precision for quantized inputs
        ALLOW_TF32: If True (and not quantized), allow TF32 precision
    
    Returns:
        Updated accumulator
    """
    return acc + (tl.dot(a, b, input_precision="ieee") if QUANTIZED
                  else tl.dot(a, b, allow_tf32=ALLOW_TF32))


@triton.jit
def load(A_BASE, B_BASE,
         stride_ak: tl.constexpr, stride_bk: tl.constexpr,
         CACHE_MODIFIER_A: tl.constexpr, CACHE_MODIFIER_B: tl.constexpr):
    """
    Load a block of A and B matrices with appropriate alignment hints.
    
    Args:
        A_BASE: Base pointer for A block
        B_BASE: Base pointer for B block
        stride_ak: Stride of A in K dimension
        stride_bk: Stride of B in K dimension
        CACHE_MODIFIER_A: Cache modifier for A loads
        CACHE_MODIFIER_B: Cache modifier for B loads
    
    Returns:
        Tuple of (a, b) loaded blocks
    """
    if stride_ak == 1:
        a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_A)
    else:
        a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_A)

    if stride_bk == 1:
        b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_B)
    else:
        b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_B)
    
    return a, b
