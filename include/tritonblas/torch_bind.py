
import functools
import itertools
import torch
from torch.library import triton_op
import triton
import triton.language as tl
from typing import Optional

import origami

from .internal.persistent_matmul import persistent_matmul


lib = torch.library.Library('tritonblas', 'FRAGMENT')


def _attype_to_origami_dtype(attype: torch.dtype) -> origami.data_type_t:
    dtype = None
    if attype is torch.float:
        dtype = origami.data_type_t.Float
    elif attype is torch.double:
        dtype = origami.data_type_t.Double
    elif attype is torch.cfloat:
        dtype = origami.data_type_t.ComplexFloat
    elif attype is torch.cdouble:
        dtype = origami.data_type_t.ComplexDouble
    elif attype is torch.half:
        dtype = origami.data_type_t.Half
    elif attype is torch.int32:
        dtype = origami.data_type_t.Int32
    elif attype is torch.bfloat16:
        dtype = origami.data_type_t.BFloat16
    elif attype is torch.int8:
        dtype = origami.data_type_t.Int8
    elif attype is torch.int64:
        dtype = origami.data_type_t.Int64

    #TODO: this is based on the old map in ATen which was more limited,
    # torch.dtype space supports more and this table should capture that
    torch._assert(dtype != None, 'Unsupported origami datatype')
    return dtype


def _generate_problem(a: torch.Tensor,
                      b: torch.Tensor,
                      c: torch.Tensor) -> origami.problem_t:
    # Initialize blank origami problem
    problem = origami.problem_t()    

    # Problem dimensions - M and K from a, N from b
    problem.size = origami.dim3_t(a.size()[0],
                                  b.size()[1],
                                  a.size()[1])

    # Batch - set to 1
    #TODO: provide knob for this?
    problem.batch = 1

    # Transpose type is TN
    #TODO: get that from the FX graph and dynamically set?
    problem.transpose_a = origami.transpose_t.T
    problem.transpose_b = origami.transpose_t.N

    # Data types for a, b, c, and output d
    problem.a_dtype = _attype_to_origami_dtype(a.dtype)
    problem.b_dtype = _attype_to_origami_dtype(b.dtype)
    problem.c_dtype = _attype_to_origami_dtype(c.dtype)
    problem.d_dtype = problem.c_dtype

    # Compute type based on a
    #TODO: provide knob for this?
    problem.mi_dtype = problem.a_dtype

    # MX block size - MX not well supported by torch at the moment
    #TODO: add support for that when torch catches up
    problem.a_mx_block_size = 0
    problem.b_mx_block_size = 0

    return problem


@functools.lru_cache(maxsize=16)
def _fetch_hardware_cached(device_id: int) -> origami.hardware_t:
    return origami.get_hardware_for_device(device_id)


#TODO: can't LRU cache this right now because lists aren't hashable
def _infer_mi_dimensions(element_bitwidth_a: int,
                         element_bitwidth_b: int,
                         n_cu: int,
                         block_mn_range: [int],
                         block_k_range: [int]) -> [int]:
    mi_dim = []
    max_bitsize = max(element_bitwidth_a, element_bitwidth_b)

    # gfx950
    if n_cu == 256:
        # FP32
        if max_bitsize == 32:
            mi_dim = [16, 16, 4]
        # FP16/BF16
        elif max_bitsize == 16:
            mi_dim = [16, 16, 32]
        # F4/F6/F8
        elif max_bitsize <= 8:
            mi_dim = [16, 16, 128]
    # gfx942 (MI300X)
    elif n_cu == 304:
        # FP32
        if max_bitsize == 32:
            mi_dim = [16, 16, 4]
        # FP16/BF16
        elif max_bitsize == 16:
            mi_dim = [16, 16, 16]
        # FP8
        elif max_bitsize == 8:
            mi_dim = [16, 16, 32]
            block_mn_range.extend([512])
            block_k_range.extend([128, 256])
        # FP4/FP6
        elif max_bitsize < 8:
            torch._assert(False, 'MI300X doesn\'t support F4/F6')
    # gfx942 (MI300A)
    elif n_cu == 228:
        # FP32
        if max_bitsize == 32:
            mi_dim = [16, 16, 4]
        # FP16/BF16
        elif max_bitsize == 16:
            mi_dim = [16, 16, 16]
        # FP8
        elif max_bitsize == 8:
            mi_dim = [16, 16, 32]
            block_mn_range.extend([512])
            block_k_range.extend([128, 256])
        # FP4/FP6
        elif max_bitsize < 8:
            torch._assert(False, 'MI300A doesn\'t support F4/F6')
    # gfx90a (MI200)
    elif n_cu == 104:
        # FP32
        if max_bitsize == 32:
            mi_dim = [16, 16, 4]
        # FP16/BF16
        elif max_bitsize == 16:
            mi_dim = [16, 16, 16]
        # FP8
        elif max_bitsize == 8:
            torch._assert(False, 'MI200 doesn\'t support F8')
        # FP4/FP6
        elif max_bitsize < 8:
            torch._assert(False, 'MI200 doesn\'t support F4/F6')

    torch._assert(mi_dim, 'No valid Matrix Instruction integrated for give datatypes')
    return mi_dim


#TODO: can't LRU cache this right now because lists aren't hashable
def _generate_config_permutations(block_mn_range: [int],
                                  block_k_range: [int],
                                  wgm_range: [int],
                                  kernel_occupancy_range: [int],
                                  mi_dim: [int]) -> [origami.config_t]:
    config_list = []

    # Combination generator based on all input parameters
    combo_gen = itertools.product(block_mn_range,
                                  block_mn_range,
                                  block_k_range,
                                  wgm_range,
                                  kernel_occupancy_range,
                                  [mi_dim[0]],
                                  [mi_dim[1]],
                                  [mi_dim[2]])

    for combo in combo_gen:
        # Unzip each combo into parts
        blk_m, blk_n, blk_k, wgm, occupancy, mi_m, mi_n, mi_k = combo

        # Store parts into a new config
        new_config = origami.config_t()
        new_config.mt = origami.dim3_t(blk_m, blk_n, blk_k)
        new_config.mi = origami.dim3_t(mi_m, mi_n, mi_k)
        new_config.workgroup_mapping = wgm
        new_config.occupancy = occupancy

        # Save config as candidate for selection
        config_list.append(new_config)

    return config_list


@triton_op('tritonblas::matmul', mutates_args={})
def _wrap_tritonblas_matmul(a: torch.Tensor,
                            b: torch.Tensor,
                            c: torch.Tensor,
                            enable_streamk: bool=False,
                            sk_grid: Optional[int]=None) -> torch.Tensor:
    # Create the problem we're solving based on the input tensors
    problem = _generate_problem(a, b, c)

    # Fetch hardware info for the device the tensors are resident on
    #TODO: assert all tensors are on the same device or let torch handle that?
    hardware = _fetch_hardware_cached(a.get_device())

    # Populate lists of valid parameters
    element_bitwidth_a = origami.datatype_to_bits(problem.a_dtype)
    element_bitwidth_b = origami.datatype_to_bits(problem.b_dtype)
    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64]
    wgm_range = [1, 2, 4, 6, 8]
    kernel_occupancy_range = [1]
    mi_dim = _infer_mi_dimensions(element_bitwidth_a,
                                  element_bitwidth_b,
                                  hardware.N_CU,
                                  block_mn_range,
                                  block_k_range)

    # Build list of candidate configs
    configs = _generate_config_permutations(block_mn_range,
                                            block_k_range,
                                            wgm_range,
                                            kernel_occupancy_range,
                                            mi_dim)

    # Run Origami prediction
    result = origami.select_config(problem, hardware, configs)

    #TODO: move above to separate class a la current tritonBLAS methodology
    #####

    blk_m = result.config.mt.m
    blk_n = result.config.mt.n
    blk_k = result.config.mt.k
    gsize_m = result.config.workgroup_mapping

    M, K = a.shape
    _, N = b.shape

    total_blocks_M = triton.cdiv(M, blk_m)
    total_blocks_N = triton.cdiv(N, blk_n)
    total_tiles = total_blocks_M * total_blocks_N
    total_programs = total_tiles
    even_k = K % blk_k == 0

    #TODO: get these values from not_a_hardcode?
    num_stages = 2
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 1
    CACHE_MODIFIER_A = None
    CACHE_MODIFIER_B = None

    num_xcds = 8
    chunk_size = gsize_m * gsize_m
    chunk_size = min(chunk_size, total_programs // num_xcds)

    grids = total_tiles

    # Torch.compile isn't too happy about aliasing outputs with inputs, so
    # we'll make a new tensor to return the output for now
    #TODO: don't do that, or at least provide the option not to
    d = c.clone()

    torch.library.wrap_triton(persistent_matmul)[(grids,)](
        a,
        b,
        d, # Would be c for in-place accumulation
        None,
        None,
        None, # No bias support at the moment
        M,
        N,
        K,
        a.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        0, # No bias support at the moment
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        BLOCK_SIZE_M=blk_m,
        BLOCK_SIZE_N=blk_n,
        BLOCK_SIZE_K=blk_k,
        GROUP_SIZE_M=gsize_m,
        NUM_SMS=total_programs,
        NUM_XCDS=num_xcds, #TODO: query this from hardware
        CHUNK_SIZE=chunk_size,
        BIAS=False,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=CACHE_MODIFIER_A,
        CACHE_MODIFIER_B=CACHE_MODIFIER_B,
        QUANTIZED=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        num_stages=num_stages,
        num_warps=num_warps,
        matrix_instr_nonkdim=mfmaInstrSize,
        kpack=kpack
    )

    return d


def _tritonblas_matmul_setup_context(ctx, inputs, output):
    # Save tensors needed for backward pass
    a, b, c, enable_streamk, sk_grid = inputs
    ctx.save_for_backward(a, b)


def _tritonblas_matmul_backward(ctx, grad_output):
    #FIXME: this is a very rough autograd implementation just so things work
    # Retrieve saved tensors
    a, b = ctx.saved_tensors
    
    # Compute gradients for each input tensor
    #TODO: use above matmul for this
    grad_a = grad_output @ b.T
    grad_b = a.T @ grad_output
    grad_c = grad_output
    
    # Return gradients for all inputs (None for non-tensor params)
    return grad_a, grad_b, grad_c, None, None


# Register autograd functionality to support backwards tritonBLAS calls
torch.library.register_autograd('tritonblas::matmul',
                                _tritonblas_matmul_backward,
                                setup_context=_tritonblas_matmul_setup_context)

