# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Matrix view aggregates for tritonblas shards.

Provides :class:`InputView`, :class:`OutputView`, :class:`ScaleView`, and 
:class:`BiasView` aggregates that encapsulate matrix pointers, dimensions, 
and strides. The kernel writer just describes their matrices and uses them - 
no need to worry about layout flags or transpose.

Example
-------

.. code-block:: python

    # A [M, K] with strides stride_am, stride_ak
    tensorA = make_tensor_view(A, M, K, stride_am, stride_ak)
    
    # B [K, N] with strides stride_bk, stride_bn
    tensorB = make_tensor_view(B, K, N, stride_bk, stride_bn)
    
    # C [M, N] with strides stride_cm, stride_cn
    tensorC = make_output_view(C, M, N, stride_cm, stride_cn)
    
    # Optional: Scale and bias views for quantized GEMM epilogue
    scale_view = make_scale_view(A_scale_ptr, B_scale_ptr, M, N)
    bias_view = make_bias_view(bias_ptr, M, stride_bias)
    
    # Use them - store handles scaling, bias, and type conversion
    acc = ctx.reduce_axis(tensorA, tensorB, out_tile)
    tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view)
"""

import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from .tile import Tile


@aggregate
class InputView:
    """
    Input matrix view for GEMM.
    
    Stores the matrix pointer, dimensions, and both strides.
    The ``tile_ptrs()`` method computes pointers using the general formula
    that works for any memory layout.
    
    Attributes:
        ptr: Base pointer to matrix data
        rows: Number of rows
        cols: Number of columns
        stride_row: Stride when moving along rows (first dimension)
        stride_col: Stride when moving along columns (second dimension)
    """
    ptr: tl.tensor
    rows: tl.tensor
    cols: tl.tensor
    stride_row: tl.tensor
    stride_col: tl.tensor
    
    @triton.constexpr_function
    def __init__(self, ptr, rows, cols, stride_row, stride_col):

        self.ptr = ptr
        self.rows = rows
        self.cols = cols
        self.stride_row = stride_row
        self.stride_col = stride_col
    
    @triton.jit
    def tile_ptrs(self, tile: Tile):
        """
        Compute pointer array and bounds mask for a tile.
        
        Uses the general formula: ptr[i,j] = base + i*stride_row + j*stride_col
        This works for any memory layout (row-major, col-major, or other).
        
        Args:
            tile: Tile object with (pid_row, pid_col, block_row, block_col)
            
        Returns:
            ptrs: 2D pointer array [BLOCK_ROW, BLOCK_COL]
            mask: 2D boolean mask for boundary handling
        """
        tl.assume(self.stride_row > 0)
        tl.assume(self.stride_col > 0)
        r_row, r_col, mask = tile.layout(self.rows, self.cols)
        ptrs = self.ptr + r_row[:, None] * self.stride_row + r_col[None, :] * self.stride_col
        return ptrs, mask
    
    @triton.jit
    def load(self, tile: Tile, boundary: tl.constexpr = False, cache_modifier: tl.constexpr = ".cg"):
        """
        Load a tile from this matrix.
        
        Args:
            tile: Tile with coordinates and shape
            boundary: If True, apply boundary masking for partial tiles
            cache_modifier: Cache modifier for load instruction
        
        Returns:
            Loaded tile data [BLOCK_ROW, BLOCK_COL]
        """
        ptrs, mask = self.tile_ptrs(tile)
        if boundary:
            return tl.load(ptrs, mask=mask, other=0.0, cache_modifier=cache_modifier)
        else:
            return tl.load(ptrs, cache_modifier=cache_modifier)


@aggregate
class ScaleView:
    """
    Scale vectors view for quantized GEMM epilogue.
    
    Stores pointers to per-row A scales and per-column B scales,
    along with dimensions for bounds checking.
    
    Attributes:
        a_scale_ptr: Pointer to A scale vector (per-row, length M)
        b_scale_ptr: Pointer to B scale vector (per-column, length N)
        M: Number of rows (for A scale bounds)
        N: Number of columns (for B scale bounds)
        stride_a: Stride for A scales (default: 1)
        stride_b: Stride for B scales (default: 1)
    """
    a_scale_ptr: tl.tensor
    b_scale_ptr: tl.tensor
    M: tl.tensor
    N: tl.tensor
    stride_a: tl.tensor
    stride_b: tl.tensor
    
    @triton.constexpr_function
    def __init__(self, a_scale_ptr, b_scale_ptr, M, N, stride_a, stride_b):

        self.a_scale_ptr = a_scale_ptr
        self.b_scale_ptr = b_scale_ptr
        self.M = M
        self.N = N
        self.stride_a = stride_a
        self.stride_b = stride_b
    
    @triton.jit
    def apply(self, acc, tile: Tile):
        """
        Apply quantization scales to accumulator.
        
        Args:
            acc: Accumulator tensor [BLOCK_M, BLOCK_N]
            tile: Tile with coordinates for indexing
        
        Returns:
            Scaled accumulator as float32
        """
        tl.assume(self.stride_a > 0)
        tl.assume(self.stride_b > 0)
        
        rm, rn = tile.indices()
        a_scales = tl.load(self.a_scale_ptr + rm * self.stride_a, mask=rm < self.M, other=1.0)
        b_scales = tl.load(self.b_scale_ptr + rn * self.stride_b, mask=rn < self.N, other=1.0)
        acc = acc.to(tl.float32)
        acc = acc * a_scales[:, None]
        acc = acc * b_scales[None, :]
        return acc


@aggregate
class BiasView:
    """
    Bias vector view for GEMM epilogue.
    
    Stores pointer to bias vector and dimension for bounds checking.
    
    Attributes:
        ptr: Pointer to bias vector (length M, broadcast across columns)
        M: Number of rows (for bounds checking)
        stride: Stride for bias vector (default: 1)
    """
    ptr: tl.tensor
    M: tl.tensor
    stride: tl.tensor
    
    @triton.constexpr_function
    def __init__(self, ptr, M, stride):
        self.ptr = ptr
        self.M = M
        self.stride = stride
    
    @triton.jit
    def apply(self, acc, tile: Tile):
        """
        Add bias vector to accumulator.
        
        Args:
            acc: Accumulator tensor [BLOCK_M, BLOCK_N]
            tile: Tile with coordinates for indexing
        
        Returns:
            Accumulator with bias added
        """
        rm, _ = tile.indices()
        bias_vector = tl.load(self.ptr + rm * self.stride, mask=rm < self.M, other=0.0)
        acc = acc + bias_vector[:, None]
        return acc


@aggregate
class OutputView:
    """
    Output matrix view for GEMM.
    
    Same design as :class:`InputView` - stores pointer, dimensions, and strides.
    Provides ``store()`` with optional epilogue (scaling, bias, type conversion).
    
    The ``store()`` method can optionally apply:
    
    * Quantization scales (from :class:`ScaleView`)
    * Bias addition (from :class:`BiasView`)
    * Type conversion to output dtype
    
    Example
    -------
    
    .. code-block:: python
    
        tensorC = make_output_view(C, M, N, stride_cm, stride_cn)
        scale_view = make_scale_view(A_scale, B_scale, M, N)
        bias_view = make_bias_view(bias, M, stride_bias)
        
        # Full epilogue: scale -> bias -> convert -> store
        tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view)
    """
    ptr: tl.tensor
    rows: tl.tensor
    cols: tl.tensor
    stride_row: tl.tensor
    stride_col: tl.tensor
    
    @triton.constexpr_function
    def __init__(self, ptr, rows, cols, stride_row, stride_col):
        self.ptr = ptr
        self.rows = rows
        self.cols = cols
        self.stride_row = stride_row
        self.stride_col = stride_col
    
    @triton.jit
    def tile_ptrs(self, tile: Tile):
        """
        Compute pointer array and bounds mask for a tile.
        """
        tl.assume(self.stride_row > 0)
        tl.assume(self.stride_col > 0)
        r_row, r_col, mask = tile.layout(self.rows, self.cols)
        ptrs = self.ptr + r_row[:, None] * self.stride_row + r_col[None, :] * self.stride_col
        return ptrs, mask
    
    @triton.jit
    def store(self, data, tile: Tile, mask=None, scale: ScaleView = None, bias: BiasView = None):
        """
        Store data to a tile with optional epilogue operations.
        
        Applies epilogue in order: scale -> bias -> type convert -> store
        
        Args:
            data: Data to store [BLOCK_ROW, BLOCK_COL]
            tile: Tile with coordinates and shape
            mask: Optional mask (if None, computes from bounds)
            scale: Optional ScaleView for quantization scaling
            bias: Optional BiasView for bias addition
        
        Example::
        
            # Simple store (no epilogue)
            tensorC.store(acc.to(C.type.element_ty), out_tile)
            
            # With full epilogue
            tensorC.store(acc, out_tile, scale=scale_view, bias=bias_view)
        """
        result = data
        
        # Apply quantization scales if provided
        if scale is not None:
            result = scale.apply(result, tile)
        
        # Add bias if provided
        if bias is not None:
            result = bias.apply(result, tile)
        
        # Type conversion to output dtype
        result = result.to(self.ptr.type.element_ty)
        
        # Compute pointers and store
        ptrs, bounds_mask = self.tile_ptrs(tile)
        if mask is None:
            mask = bounds_mask
        tl.store(ptrs, result, mask=mask)
    
    @triton.jit
    def load(self, tile: Tile, boundary: tl.constexpr = False, cache_modifier: tl.constexpr = ".cg"):
        """
        Load a tile from this matrix (for read-modify-write patterns).
        """
        ptrs, mask = self.tile_ptrs(tile)
        if boundary:
            return tl.load(ptrs, mask=mask, other=0.0, cache_modifier=cache_modifier)
        else:
            return tl.load(ptrs, cache_modifier=cache_modifier)


# =============================================================================
# Factory Functions
# =============================================================================

@triton.jit
def make_input_view(ptr, rows, cols, stride_row, stride_col) -> InputView:
    """
    Create an InputView with automatic stride type coercion.
    
    This factory ensures strides are always tensor-typed, handling the case 
    where contiguous dimensions have stride=1 (Python int) while other 
    dimensions have tensor-typed strides.
    
    Args:
        ptr: Base pointer to matrix data
        rows: Number of rows (first dimension) - must be a tensor
        cols: Number of columns (second dimension)
        stride_row: Stride when moving along rows
        stride_col: Stride when moving along columns
    
    Returns:
        InputView with all fields as tensors
    
    Example::
    
        # A is [M, K] matrix - strides can be int or tensor
        tensorA = make_input_view(A, M, K, stride_am, stride_ak)
        
        # B is [K, N] matrix
        tensorB = make_input_view(B, K, N, stride_bk, stride_bn)
    """
    # ═══════════════════════════════════════════════════════════════════════
    # TYPE PROMOTION TRICK
    # ═══════════════════════════════════════════════════════════════════════
    # Triton aggregates require strongly-typed fields (tl.tensor). However,
    # strides can be either Python ints (stride=1 for contiguous dimensions)
    # or Triton tensors (stride>1 from kernel params).
    #
    # The pattern `stride + 0 * rows` promotes any int to a tensor:
    #   - 0 * rows produces a tensor with value 0 (since rows is a tensor)
    #   - stride + (tensor 0) = tensor with stride's value
    #
    # This has ZERO runtime cost - the compiler constant-folds 0*x and x+0.
    # ═══════════════════════════════════════════════════════════════════════
    stride_row_t = stride_row + 0 * rows
    stride_col_t = stride_col + 0 * rows
    
    return InputView(ptr, rows, cols, stride_row_t, stride_col_t)


@triton.jit
def make_output_view(ptr, rows, cols, stride_row, stride_col) -> OutputView:
    """
    Create an OutputView with automatic stride type coercion.
    
    Same as make_input_view() but returns an OutputView which has
    store() method in addition to load().
    
    Args:
        ptr: Base pointer to matrix data
        rows: Number of rows (first dimension) - must be a tensor
        cols: Number of columns (second dimension)
        stride_row: Stride when moving along rows
        stride_col: Stride when moving along columns
    
    Returns:
        OutputView with all fields as tensors
    
    Example::
    
        # C is [M, N] output matrix
        tensorC = make_output_view(C, M, N, stride_cm, stride_cn)
    """
    # ═══════════════════════════════════════════════════════════════════════
    # TYPE PROMOTION TRICK - See make_input_view() for detailed explanation
    # ═══════════════════════════════════════════════════════════════════════
    stride_row_t = stride_row + 0 * rows
    stride_col_t = stride_col + 0 * rows
    
    return OutputView(ptr, rows, cols, stride_row_t, stride_col_t)


# Alias for backward compatibility
make_tensor_view = make_input_view


@triton.jit
def make_scale_view(a_scale_ptr, b_scale_ptr, M, N, stride_a=1, stride_b=1) -> ScaleView:
    """
    Create a ScaleView for quantized GEMM epilogue.
    
    Stores per-row A scales and per-column B scales with automatic
    stride type coercion.
    
    Args:
        a_scale_ptr: Pointer to A scale vector (per-row, length M)
        b_scale_ptr: Pointer to B scale vector (per-column, length N)
        M: Number of rows (for A scale bounds) - must be a tensor
        N: Number of columns (for B scale bounds)
        stride_a: Stride for A scales (default: 1)
        stride_b: Stride for B scales (default: 1)
    
    Returns:
        ScaleView with all fields as tensors
    
    Example::
    
        scale_view = make_scale_view(A_scale_ptr, B_scale_ptr, M, N)
        tensorC.store(acc, out_tile, scale=scale_view)
    """
    # Type promotion for strides
    stride_a_t = stride_a + 0 * M
    stride_b_t = stride_b + 0 * M
    
    return ScaleView(a_scale_ptr, b_scale_ptr, M, N, stride_a_t, stride_b_t)


@triton.jit
def make_bias_view(bias_ptr, M, stride=1) -> BiasView:
    """
    Create a BiasView for GEMM epilogue.
    
    Stores bias vector pointer with automatic stride type coercion.
    
    Args:
        bias_ptr: Pointer to bias vector (length M)
        M: Number of rows (for bounds checking) - must be a tensor
        stride: Stride for bias vector (default: 1)
    
    Returns:
        BiasView with all fields as tensors
    
    Example::
    
        bias_view = make_bias_view(bias_ptr, M, stride_bias)
        tensorC.store(acc, out_tile, bias=bias_view)
    """
    # Type promotion for stride
    stride_t = stride + 0 * M
    
    return BiasView(bias_ptr, M, stride_t)


# =============================================================================
# Legacy aliases for backward compatibility
# =============================================================================

A_View = InputView
B_View = InputView
C_View = OutputView
MatrixView = OutputView
InputTensorA = InputView
InputTensorB = InputView
