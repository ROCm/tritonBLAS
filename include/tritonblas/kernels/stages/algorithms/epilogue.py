"""
Epilogue functions for composable Triton GEMM kernels.
Operations applied to the output accumulator after GEMM computation.
"""
import triton
import triton.language as tl


@triton.jit
def relu(acc):
    """
    Apply ReLU activation: max(0, x)
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
    
    Returns:
        Accumulator with ReLU applied
    """
    return tl.maximum(acc, 0.0)


@triton.jit
def gelu(acc):
    """
    Apply GELU activation (approximate version using tanh).
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
    
    Returns:
        Accumulator with GELU applied
    """
    # Constants for GELU approximation
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/π)
    coeff = 0.044715
    
    # GELU approximation using numerically stable tanh
    x_cubed = acc * acc * acc
    inner = sqrt_2_over_pi * (acc + coeff * x_cubed)
    
    # Numerically stable tanh: use different formulas for positive/negative values
    # For x > 0: tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
    # For x < 0: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    exp_neg_2x = tl.exp(-2.0 * tl.abs(inner))
    tanh_inner = tl.where(
        inner >= 0,
        (1.0 - exp_neg_2x) / (1.0 + exp_neg_2x),
        -(1.0 - exp_neg_2x) / (1.0 + exp_neg_2x)
    )
    return 0.5 * acc * (1.0 + tanh_inner)


@triton.jit
def gelu_tanh(acc):
    """
    Apply GELU activation using tanh approximation (alias for gelu).
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
    
    Returns:
        Accumulator with GELU applied
    """
    return gelu(acc)


@triton.jit
def sigmoid(acc):
    """
    Apply Sigmoid activation: 1 / (1 + exp(-x))
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
    
    Returns:
        Accumulator with Sigmoid applied
    """
    return tl.sigmoid(acc)


@triton.jit
def silu(acc):
    """
    Apply SiLU (Swish) activation: x * sigmoid(x)
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
    
    Returns:
        Accumulator with SiLU applied
    """
    return acc * tl.sigmoid(acc)


@triton.jit
def tanh(acc):
    """
    Apply Tanh activation using numerically stable formula.
    For x > 0: tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
    For x < 0: tanh(x) = -(1 - exp(2x)) / (1 + exp(2x))
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
    
    Returns:
        Accumulator with Tanh applied
    """
    # Use numerically stable formula to avoid overflow
    exp_neg_2x = tl.exp(-2.0 * tl.abs(acc))
    result = (1.0 - exp_neg_2x) / (1.0 + exp_neg_2x)
    # Apply sign
    return tl.where(acc >= 0, result, -result)


@triton.jit
def leaky_relu(acc, negative_slope: tl.constexpr = 0.01):
    """
    Apply Leaky ReLU activation: max(0, x) + negative_slope * min(0, x)
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
        negative_slope: Slope for negative values (default: 0.01)
    
    Returns:
        Accumulator with Leaky ReLU applied
    """
    return tl.where(acc > 0, acc, acc * negative_slope)


@triton.jit
def identity(acc):
    """
    Identity function (no activation).
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
    
    Returns:
        Unchanged accumulator
    """
    return acc


@triton.jit
def apply_epilogue(acc, epilogue_fn):
    """
    Apply an epilogue function to the accumulator.
    
    Args:
        acc: Accumulator tensor [BLOCK_SIZE_M, BLOCK_SIZE_N]
        epilogue_fn: Epilogue function to apply (e.g., relu, gelu, etc.)
    
    Returns:
        Accumulator with epilogue applied
    """
    return epilogue_fn(acc)
