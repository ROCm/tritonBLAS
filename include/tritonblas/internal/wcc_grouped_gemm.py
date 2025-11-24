import triton
import triton.language as tl
import triton.profiler.language as pl


"""
This is a user defined function, where partial results are being stored and accumulated

Input:
start_index <- Uint    : The starting index of "work_tile" that this program will process. Positive integer
end_index <- Uint      : The last index(exclusive) of "work_tile" that this program will process. Positive integer
tile_id  <- Uint       : ID of the tile this program is processing. Positive integer
tile_offset  <- Uint   : The offset within the tile. [0, work_tile)
work_tile <- Uint      : User provides the smallest chunk/tile of work. Positive interger
NUM_PRGMS <- Uint      : Total number of programs that the kernel was launched with. Positive integer


The following inputs can be variadic arguments or arguments passed in an object by the user
However, they are pasased in as arguments here because triton does not support variadic arguments or objects
partials <- float      : This holds the partials computed by the program so far
P <- tensor            : A user defined tensor used to hold partial values
locks <- tensor        : A user defined tensor used for book keeping of accumulated partials
n_cols <- uint         : Used to compute the RMS value
BLOCK_SIZE_M <- uint   : Used to compute the size of tensor to store
BLOCK_SIZE_N <- uint   : Used to compute the size of tensor to store

Output:
partials <- tensor     : Now the accumulated tensor can be stored
"""


@triton.jit
def accumulate_partials(
    pid,
    start_index,
    end_index,
    tile_id,
    tile_offset,
    work_tile,
    partials,
    P,
    locks,
    NUM_PRGMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    streamk_tiles_pcu: tl.constexpr,
    streamk_remainder_tiles: tl.constexpr,
):
    if tile_offset != 0:
        rm1 = tl.arange(0, BLOCK_SIZE_M)
        rn1 = tl.arange(0, BLOCK_SIZE_N)
        rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
        P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
        tl.store(P_, partials, cache_modifier=".wt")
        tl.store(locks + pid, 1, cache_modifier=".wt")
    # Only the pid processing the first tile does the reduction
    else:
        tile_iter = tile_id * work_tile
        next_pid = pid + 1
        tile_iter_end = tile_iter + work_tile
        end = end_index
        while end < tile_iter_end and next_pid < NUM_PRGMS:
            while tl.load(locks + next_pid, cache_modifier=".cv", volatile=True) != 1:
                pass
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            partials += tl.load(tl.multiple_of(P_, (1, 16)), cache_modifier=".cv")

            next_pid_start_index = next_pid * streamk_tiles_pcu + tl.minimum(next_pid, streamk_remainder_tiles)
            next_pid_end_index = (next_pid + 1) * streamk_tiles_pcu + tl.minimum(next_pid + 1, streamk_remainder_tiles)
            next_pid_curr_gemm_start = tl.maximum(next_pid_start_index, tile_iter)
            next_pid_curr_gemm_end = tl.minimum(next_pid_end_index, tile_iter_end)
            num_tiles = next_pid_curr_gemm_end - next_pid_curr_gemm_start

            if num_tiles > 0:
                end += num_tiles
            next_pid += 1

    return partials


"""
The function calculates calculates indices needed for each iteration. Specifically it calculates the end index of this
iteration, tile id of the tile being processed, and the offset within that tile

Input:
start_index <- Uint    : The starting index of "work_tile" that this program will process. Positive integer
last_index <- Uint     : The last index(exclusive) of "work_tile" that this program will process. Positive integer
work_tile <- Uint      : User provides the smallest chunk/tile of work. Positive interger

Output:
end_index  <- Uint      : The index of the last "atomic tile" of this iteration
tile_id  <- Uint        : Returns which tile this program is processing
tile_offset  <- Uint    : Returns the offset within a tile. [0, work_tile)
"""


@triton.jit
def per_iter_indices(start_index, last_index, work_tile):
    tile_offset = start_index % work_tile
    end_index = tl.minimum(start_index + (work_tile - tile_offset), last_index)
    tile_id = start_index // work_tile
    return (end_index, tile_id, tile_offset)


"""
Given the total streamk_tiles_pcu and streamk_remainder_tiles, the function computes the first and the last index of the work_iles
that the given pid will process.
Inherently the function is splitting the work evenly among the pids

Input:
pid <- Uint                     : User provides the PID of the program. Positive interger in the range of [0:NUM_PRGMS)  #Assuming 1D grid launch
streamk_tiles_pcu <- Uint       : Total number of tiles per CU
streamk_remainder_tiles <- Uint : Remainder number of tiles
NUM_PRGMS <- Uint               : Total number of programs that the kernel was launched with. It is needed to split the work. Positive integer

Output:
start_index <- Uint    : The starting index of "work_tile" that this program will process. Positive integer
last_index <- Uint     : The last index(exclusive) of "work_tile" that this program will process. Positive integer
"""


@triton.jit
def work_split(
    pid,
    streamk_tiles_pcu: tl.constexpr,
    streamk_remainder_tiles: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    start_index = pid * streamk_tiles_pcu + tl.minimum(pid, streamk_remainder_tiles)
    last_index = (pid + 1) * streamk_tiles_pcu + tl.minimum(pid + 1, streamk_remainder_tiles)
    return (start_index, last_index)


"""
Work Centric Grouped GEMM implementation

Inputs:
group_a_ptrs <- pointer     : A pointer which points to all the 'A' matrices
group_b_ptrs <- pointer     : A pointer which points to all the 'C' matrices
group_c_ptrs <- pointer     : A pointer which points to all the 'C' matrices
group_gemm_sizes <- pointer : A pointer which points to all the matrix sizes: [m, n, k]
gemm_offsets <- tensor      : This tensor is essentially a inclusive prefix sum array of all
                              the linearized tiles of all the gemms. It is needed for each pid
                              to know which gemm should it be processing
g_lds <- pointer            : A pointer which points to all the stride values for each matrix
                              [A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1)]
group_size <- uint          : Total number of gemm we have
P <- tensor                 : A user defined tensor used to hold partial values
locks <- tensor             : A user defined tensor used for book keeping of accumulated partials
streamk_tiles_pcu <- uint   : Total number of tiles per CU
streamk_remainder_tiles <- uint : Remainder number of tiles
BLOCK_SIZE_M <- uint        : Block size in the 'm' dimension
BLOCK_SIZE_N <- uint        : Block size in the 'n' dimension
BLOCK_SIZE_K <- uint        : Block size in the 'k' dimension
GROUP_SIZE_M <- uint        : To assign work in a more lds friendly manner
NUM_PRGMS <- uint           : The number of program the kernel was launched with
NUM_XCDS <- uint            : Total number of XCDS in the hardware

Output
group_c_ptrs <- pointer     : A pointer which points to all the 'C' matrices (now populated)
"""


@triton.jit()
def wcc_groupgemm(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,
    gemm_offsets,
    g_lds,
    group_size,
    P,
    locks,
    streamk_tiles_pcu: tl.constexpr,
    streamk_remainder_tiles: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_PRGMS // NUM_XCDS) + (pid // NUM_XCDS)

    # Calculate the start and last tile this pid would be processing. This is linearized tiles
    # for all the gemms
    start_index, last_iter = work_split(pid, streamk_tiles_pcu, streamk_remainder_tiles, NUM_PRGMS)
    for g in range(group_size):
        # Check to see if this pid needs to process the "g th" gemm
        g_val = tl.load(gemm_offsets + g + 1)
        if start_index < g_val and start_index != last_iter:
            # If it does, find the end of that gemm
            last_index = tl.minimum(last_iter, g_val)
            # Core loop
            while start_index < last_index:
                # Load in all the corresponding data for that gemm
                M = tl.load(group_gemm_sizes + g * 3)
                N = tl.load(group_gemm_sizes + g * 3 + 1)
                K = tl.load(group_gemm_sizes + g * 3 + 2)

                A = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
                B = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
                C = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))

                stride_am = tl.load(g_lds + g * 6)
                stride_ak = tl.load(g_lds + g * 6 + 1)
                stride_bk = tl.load(g_lds + g * 6 + 2)
                stride_bn = tl.load(g_lds + g * 6 + 3)
                stride_cm = tl.load(g_lds + g * 6 + 4)
                stride_cn = tl.load(g_lds + g * 6 + 5)

                num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
                num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
                work_tile = tl.cdiv(K, BLOCK_SIZE_K)

                acc_dtype = tl.float32  # if C.type.element_ty != tl.int8 else tl.int32
                end_index, tile_id, tile_offset = per_iter_indices(start_index, last_index, work_tile)

                pid_m = tile_id // num_pid_n
                pid_n = tile_id % num_pid_n
                tl.assume(pid_m >= 0)
                tl.assume(pid_n >= 0)

                rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
                rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
                rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
                rk = tl.arange(0, BLOCK_SIZE_K)

                """
                The following two lines, support all transpose types, however the triton compiler is unable to optimize
                it, leading to 'short' loads rather than 'dwordx4' loads
                A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
                B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder
                """
                A_BASE = A + rm[:, None] * stride_am + rk[None, :] + (BLOCK_SIZE_K * tile_offset)
                B_BASE = B + rk[:, None] * stride_bk + rn[None, :] + (BLOCK_SIZE_K * stride_bk * tile_offset)
                partials = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
                for current_iter in range(start_index, end_index):
                    """
                    The following masking logic is omitted because it leads to 'short' loads rather than 'dwordx4' loads
                    However, it has been tested and can be added back anytime
                    """
                    # global_k_offset = (current_iter % work_tile) * BLOCK_SIZE_K
                    # mask = global_k_offset + rk < K
                    A_BASE = tl.multiple_of(A_BASE, (16, 16))
                    B_BASE = tl.multiple_of(B_BASE, (16, 16))
                    a = tl.load(A_BASE)
                    b = tl.load(B_BASE)
                    # do the actual gemm computation
                    partials += tl.dot(a, b)
                    # The following line has been omitted to make sure loads are 'dwordx4'
                    # A_BASE += BLOCK_SIZE_K * stride_ak
                    A_BASE += BLOCK_SIZE_K
                    B_BASE += BLOCK_SIZE_K * stride_bk

                # work_fixup()
                partials = accumulate_partials(
                    pid,
                    start_index,
                    end_index,
                    tile_id,
                    tile_offset,
                    work_tile,
                    partials,
                    P,
                    locks,
                    NUM_PRGMS,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    streamk_tiles_pcu,
                    streamk_remainder_tiles,
                )
                # Only the pid which does the first chunk of the tiles, stores the whole tile to output
                if tile_offset == 0:
                    c = partials.to(C.type.element_ty)
                    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
                    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
                    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
                    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
                    C_ = C + rm[:, None] * stride_cm + rn[None, :]
                    """
                    The following two lines are omitted because they cause the stores to be 'short' rather than
                    'dwordx2', however, they are tested and can be added back anytime
                    C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
                    mask = (rm < M)[:, None] & (rn < N)[None, :]
                    """
                    C_ = tl.multiple_of(C_, (16, 16))
                    tl.store(C_, c)

                start_index = end_index


"""
The following function was adapted from https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html
This function is used as a reference to check correctness of WCC grouped gemm. It only supports square matrices
and only one transpose type
"""


@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles
