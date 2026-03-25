#!/usr/bin/env python3
"""
Collect MALL, L2, HBM counters for GEMM alone vs GEMM+RCCL overlap.

Standalone script — run inside the mall-experiments container with rocprofv3.
Does NOT use rocprofv3 wrapper; instead uses PyTorch + triton directly and
reads hardware counters via rocprofv3 wrapping the entire process.

This script runs ONE scenario at a time (controlled by --scenario):
  - torch_alone:  torch.matmul in isolation
  - ws_alone:     WS Hierarchical in isolation
  - torch_rccl:   torch.matmul overlapped with RCCL all_reduce
  - ws_rccl:      WS Hierarchical overlapped with RCCL all_reduce

For RCCL scenarios, launch with torchrun --nproc_per_node=8.
For alone scenarios, launch with python3 directly (single GPU).

Usage:
  # Alone (single GPU):
  rocprofv3 --pmc <COUNTERS> -o /tmp/out -- python3 benchmarks/collect_counters.py --scenario torch_alone
  rocprofv3 --pmc <COUNTERS> -o /tmp/out -- python3 benchmarks/collect_counters.py --scenario ws_alone

  # RCCL overlap (8 GPUs):
  rocprofv3 --pmc <COUNTERS> -o /tmp/out -- torchrun --nproc_per_node=8 benchmarks/collect_counters.py --scenario torch_rccl
  rocprofv3 --pmc <COUNTERS> -o /tmp/out -- torchrun --nproc_per_node=8 benchmarks/collect_counters.py --scenario ws_rccl
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

# Stub out tritonblas package to avoid origami import
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

N_CU = 304
NUM_XCDS = 8
BLK_M = 256; BLK_N = 256; BLK_K = 64
GROUP_SIZE_M = 8
COUNTER_STRIDE = 64
SIZE = 8192
COMM_SIZE = 16384
WARMUP = 5
STEPS = 20
dtype = torch.bfloat16


def hierarchical_split(M, N):
    total_tiles = triton.cdiv(M, BLK_M) * triton.cdiv(N, BLK_N)
    tiles_per_cu = total_tiles / max(N_CU, 1)
    local_frac = max(0.5, 1.0 - max(0.0, tiles_per_cu - 4.0) * 0.05)
    local_per_xcd = int(total_tiles * local_frac) // NUM_XCDS
    local_per_xcd = max(local_per_xcd, 1)
    global_tiles = total_tiles - local_per_xcd * NUM_XCDS
    return local_per_xcd, global_tiles


def make_ws(A, B, C):
    M, K = A.shape; _, N = B.shape; dev = A.device
    even_k = K % BLK_K == 0
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
            LOCAL_TILES_PER_XCD=lp, GLOBAL_TILES=gt,
            COUNTER_STRIDE=COUNTER_STRIDE,
            BIAS=False, EVEN_K=even_k,
            CACHE_MODIFIER_A=None, CACHE_MODIFIER_B=None, QUANTIZED=False,
            num_stages=2, num_warps=8, waves_per_eu=0,
            matrix_instr_nonkdim=16, kpack=1,
            USE_MASK=True, mask_ptr=mask,
        )
    def reset(): tc.zero_(); gc.zero_()
    return fn, reset


def run_alone(scenario):
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    M = N = K = SIZE
    A = torch.randn(M, K, dtype=dtype, device=dev)
    B = torch.randn(K, N, dtype=dtype, device=dev)

    if scenario == "torch_alone":
        fn = lambda: torch.matmul(A, B)
        reset = lambda: None
    else:
        C = torch.zeros(M, N, dtype=dtype, device=dev)
        fn, reset = make_ws(A, B, C)

    for _ in range(WARMUP):
        reset(); fn()
    torch.cuda.synchronize()

    for i in range(STEPS):
        reset()
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()

    print(f"[{scenario}] Completed {STEPS} iterations on GPU 0", flush=True)


def run_rccl(scenario):
    import torch.distributed as dist
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)

    M = N = K = SIZE
    A = torch.randn(M, K, dtype=dtype, device=dev)
    B = torch.randn(K, N, dtype=dtype, device=dev)
    comm_buf = torch.randn(COMM_SIZE, COMM_SIZE, dtype=dtype, device=dev)

    mm_stream = torch.cuda.Stream(device=dev)
    co_stream = torch.cuda.Stream(device=dev)

    if scenario == "torch_rccl":
        fn = lambda: torch.matmul(A, B)
        reset = lambda: None
    else:
        C = torch.zeros(M, N, dtype=dtype, device=dev)
        fn, reset = make_ws(A, B, C)

    for _ in range(WARMUP):
        reset()
        with torch.cuda.stream(co_stream): dist.all_reduce(comm_buf)
        with torch.cuda.stream(mm_stream): fn()
    torch.cuda.synchronize()

    for i in range(STEPS):
        reset()
        torch.cuda.synchronize()
        with torch.cuda.stream(co_stream): dist.all_reduce(comm_buf)
        with torch.cuda.stream(mm_stream): fn()
        torch.cuda.synchronize()

    if rank == 0:
        print(f"[{scenario}] Completed {STEPS} iterations on 8 GPUs", flush=True)

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True,
                        choices=["torch_alone", "ws_alone", "torch_rccl", "ws_rccl"])
    args = parser.parse_args()

    if args.scenario.endswith("_rccl"):
        run_rccl(args.scenario)
    else:
        run_alone(args.scenario)


if __name__ == "__main__":
    main()
