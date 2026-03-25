#!/usr/bin/env python3
"""
Collect ALL MALL + HBM counters for GEMM alone vs GEMM+RCCL overlap.
Simplified: single GPU for alone, 8 GPU for RCCL.
Each run does a single counter group to stay within rocprofv3 limits.
"""
import argparse, importlib.util, os, sys, time, types
import torch
import triton
import triton.language as tl

_inc = os.path.join(os.path.dirname(__file__), "..", "include")

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

pkg = types.ModuleType("tritonblas")
pkg.__path__ = [os.path.join(_inc, "tritonblas")]
sys.modules["tritonblas"] = pkg
for sub in ["kernels", "kernels.stages", "kernels.stages.indexing"]:
    full = f"tritonblas.{sub}"
    m = types.ModuleType(full)
    m.__path__ = [os.path.join(_inc, "tritonblas", *sub.split("."))]
    sys.modules[full] = m

_load_module(
    "tritonblas.kernels.stages.indexing.pid_transforms",
    os.path.join(_inc, "tritonblas", "kernels", "stages", "indexing", "pid_transforms.py"),
)
_kern = _load_module(
    "tritonblas.kernels.persistent_gemm_ws_hierarchical",
    os.path.join(_inc, "tritonblas", "kernels", "persistent_gemm_ws_hierarchical.py"),
)
ws_hierarchical_matmul = _kern.ws_hierarchical_matmul

N_CU = 304; NUM_XCDS = 8; BLK_M = 256; BLK_N = 256; BLK_K = 64
GROUP_SIZE_M = 8; COUNTER_STRIDE = 64
SIZE = 8192; COMM_SIZE = 16384
WARMUP = 5; STEPS = 30
dtype = torch.bfloat16


def hierarchical_split(M, N):
    total_tiles = triton.cdiv(M, BLK_M) * triton.cdiv(N, BLK_N)
    tiles_per_cu = total_tiles / max(N_CU, 1)
    local_frac = max(0.5, 1.0 - max(0.0, tiles_per_cu - 4.0) * 0.05)
    local_per_xcd = int(total_tiles * local_frac) // NUM_XCDS
    local_per_xcd = max(local_per_xcd, 1)
    return local_per_xcd, total_tiles - local_per_xcd * NUM_XCDS


def make_ws(A, B, C):
    M, K = A.shape; _, N = B.shape; dev = A.device
    lp, gt = hierarchical_split(M, N)
    tc = torch.zeros(NUM_XCDS * COUNTER_STRIDE, device=dev, dtype=torch.int32)
    gc = torch.zeros(COUNTER_STRIDE, device=dev, dtype=torch.int32)
    mask = torch.ones(N_CU, dtype=torch.int32, device=dev)
    def fn():
        ws_hierarchical_matmul[(N_CU,)](
            A, B, C, None, None, None, tc, gc,
            M, N, K, A.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
            stride_ak=A.stride(1), stride_bk=B.stride(0),
            BLOCK_SIZE_M=BLK_M, BLOCK_SIZE_N=BLK_N, BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=GROUP_SIZE_M, NUM_SMS=N_CU, NUM_XCDS=NUM_XCDS,
            LOCAL_TILES_PER_XCD=lp, GLOBAL_TILES=gt, COUNTER_STRIDE=COUNTER_STRIDE,
            BIAS=False, EVEN_K=(K % BLK_K == 0),
            CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None, QUANTIZED=False,
            num_stages=2, num_warps=8, waves_per_eu=0,
            matrix_instr_nonkdim=16, kpack=1, USE_MASK=True, mask_ptr=mask,
        )
    def reset(): tc.zero_(); gc.zero_()
    return fn, reset


def run_alone(scenario):
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    A = torch.randn(SIZE, SIZE, dtype=dtype, device=dev)
    B = torch.randn(SIZE, SIZE, dtype=dtype, device=dev)
    if scenario == "torch_alone":
        fn = lambda: torch.matmul(A, B); reset = lambda: None
    else:
        C = torch.zeros(SIZE, SIZE, dtype=dtype, device=dev)
        fn, reset = make_ws(A, B, C)
    for _ in range(WARMUP): reset(); fn()
    torch.cuda.synchronize()
    for _ in range(STEPS):
        reset(); torch.cuda.synchronize(); fn(); torch.cuda.synchronize()
    print(f"[{scenario}] {STEPS} iters GPU 0", flush=True)


def run_rccl(scenario):
    import torch.distributed as dist
    dist.init_process_group("nccl")
    rank = dist.get_rank(); dev = torch.device("cuda", rank); torch.cuda.set_device(dev)
    A = torch.randn(SIZE, SIZE, dtype=dtype, device=dev)
    B = torch.randn(SIZE, SIZE, dtype=dtype, device=dev)
    comm_buf = torch.randn(COMM_SIZE, COMM_SIZE, dtype=dtype, device=dev)
    mm_s = torch.cuda.Stream(device=dev); co_s = torch.cuda.Stream(device=dev)
    if scenario == "torch_rccl":
        fn = lambda: torch.matmul(A, B); reset = lambda: None
    else:
        C = torch.zeros(SIZE, SIZE, dtype=dtype, device=dev)
        fn, reset = make_ws(A, B, C)
    for _ in range(WARMUP):
        reset()
        with torch.cuda.stream(co_s): dist.all_reduce(comm_buf)
        with torch.cuda.stream(mm_s): fn()
    torch.cuda.synchronize()
    for _ in range(STEPS):
        reset(); torch.cuda.synchronize()
        with torch.cuda.stream(co_s): dist.all_reduce(comm_buf)
        with torch.cuda.stream(mm_s): fn()
        torch.cuda.synchronize()
    if rank == 0: print(f"[{scenario}] {STEPS} iters 8 GPUs", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True,
                   choices=["torch_alone", "ws_alone", "torch_rccl", "ws_rccl"])
    a = p.parse_args()
    (run_rccl if a.scenario.endswith("_rccl") else run_alone)(a.scenario)
