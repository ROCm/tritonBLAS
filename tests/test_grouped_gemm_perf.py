"""Performance benchmarks for grouped GEMM on MI300X.

Run with:
    python3 -m pytest tests/test_grouped_gemm_perf.py -v -s --tb=short -m performance
"""

import pytest
import torch
import tritonblas


# --- Timing helper ---

def bench_fn(fn, warmup=5, rep=20):
    """Warm up then time fn() using CUDA events. Returns median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]  # median in ms


def compute_tflops(shapes, time_ms):
    """Compute TFLOPS given a list of (M, N, K) shapes and elapsed time in ms."""
    total_flops = 2 * sum(m * n * k for m, n, k in shapes)
    time_s = time_ms / 1e3
    return total_flops / time_s / 1e12


def make_group(shapes, dtype=torch.float16):
    """Allocate A, B, C tensor lists for the given shapes."""
    group_a, group_b, group_c = [], [], []
    for m, n, k in shapes:
        group_a.append(torch.randn(m, k, device="cuda", dtype=dtype))
        group_b.append(torch.randn(k, n, device="cuda", dtype=dtype))
        group_c.append(torch.empty(m, n, device="cuda", dtype=dtype))
    return group_a, group_b, group_c


def print_table(header, rows, col_widths=None):
    """Print a simple aligned table."""
    if col_widths is None:
        col_widths = [max(len(str(r[i])) for r in [header] + rows) + 2 for i in range(len(header))]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    fmt = "|" + "|".join(f"{{:^{w}}}" for w in col_widths) + "|"
    print(sep)
    print(fmt.format(*header))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)


# --- Benchmarks ---

@pytest.mark.performance
def test_benchmark1_square_compute_bound():
    """Square compute-bound: (4096, 4096, 4096) x 4, fp16.
    Compares grouped_gemm vs torch.matmul loop."""
    shape = (4096, 4096, 4096)
    group_size = 4
    shapes = [shape] * group_size
    dtype = torch.float16

    group_a, group_b, group_c = make_group(shapes, dtype)

    def fn_grouped():
        tritonblas.grouped_gemm(group_a, group_b, group_c)

    def fn_torch():
        for a, b, c in zip(group_a, group_b, group_c):
            torch.matmul(a, b, out=c)

    t_grouped = bench_fn(fn_grouped)
    t_torch = bench_fn(fn_torch)

    tflops_grouped = compute_tflops(shapes, t_grouped)
    tflops_torch = compute_tflops(shapes, t_torch)

    header = ["Backend", "Shape", "Groups", "Time (ms)", "TFLOPS", "vs torch"]
    rows = [
        ["grouped_gemm", f"{shape[0]}x{shape[1]}x{shape[2]}", group_size,
         f"{t_grouped:.3f}", f"{tflops_grouped:.2f}", "1.00x"],
        ["torch.matmul", f"{shape[0]}x{shape[1]}x{shape[2]}", group_size,
         f"{t_torch:.3f}", f"{tflops_torch:.2f}",
         f"{tflops_grouped / tflops_torch:.2f}x"],
    ]

    print("\n\n=== Benchmark 1: Square Compute-Bound (4096x4096x4096 x4) ===")
    print_table(header, rows)
    print(f"  grouped_gemm TFLOPS: {tflops_grouped:.2f}")
    print(f"  torch loop TFLOPS:   {tflops_torch:.2f}")
    print(f"  Speedup: {tflops_grouped / tflops_torch:.3f}x")

    assert tflops_grouped > 0, "grouped_gemm must produce nonzero TFLOPS"


@pytest.mark.performance
def test_benchmark2_llm_inference_shapes():
    """LLM inference shapes (mixed M/N/K). Compares grouped_gemm vs torch loop."""
    shapes = [
        (1, 4096, 4096),
        (32, 4096, 11008),
        (1, 11008, 4096),
        (32, 11008, 4096),
    ]
    dtype = torch.float16

    group_a, group_b, group_c = make_group(shapes, dtype)

    def fn_grouped():
        tritonblas.grouped_gemm(group_a, group_b, group_c)

    def fn_torch():
        for a, b, c in zip(group_a, group_b, group_c):
            torch.matmul(a, b, out=c)

    t_grouped = bench_fn(fn_grouped)
    t_torch = bench_fn(fn_torch)

    tflops_grouped = compute_tflops(shapes, t_grouped)
    tflops_torch = compute_tflops(shapes, t_torch)

    header = ["Backend", "Time (ms)", "TFLOPS"]
    rows = [
        ["grouped_gemm", f"{t_grouped:.3f}", f"{tflops_grouped:.4f}"],
        ["torch.matmul loop", f"{t_torch:.3f}", f"{tflops_torch:.4f}"],
    ]

    print("\n\n=== Benchmark 2: LLM Inference Shapes ===")
    print("Shapes: (1,4096,4096), (32,4096,11008), (1,11008,4096), (32,11008,4096)")
    print_table(header, rows)
    print(f"  Speedup: {tflops_grouped / tflops_torch:.3f}x")

    assert tflops_grouped > 0


@pytest.mark.performance
def test_benchmark3_group_count_scaling():
    """Fixed shape (2048,2048,2048) per group, group_size in [1, 2, 4, 8, 16]."""
    base_shape = (2048, 2048, 2048)
    group_sizes = [1, 2, 4, 8, 16]
    dtype = torch.float16

    header = ["group_size", "Time (ms)", "TFLOPS"]
    rows = []

    print("\n\n=== Benchmark 3: Group Count Scaling (2048x2048x2048 per group) ===")

    for gs in group_sizes:
        shapes = [base_shape] * gs
        group_a, group_b, group_c = make_group(shapes, dtype)

        def fn_grouped(ga=group_a, gb=group_b, gc=group_c):
            tritonblas.grouped_gemm(ga, gb, gc)

        t = bench_fn(fn_grouped)
        tflops = compute_tflops(shapes, t)
        rows.append([gs, f"{t:.3f}", f"{tflops:.2f}"])

    print_table(header, rows)

    assert len(rows) == len(group_sizes)


@pytest.mark.performance
def test_benchmark4_origami_vs_fixed_tiles():
    """Origami auto-selection vs fixed BLK_M=128, BLK_N=128, BLK_K=64.
    Shape: (2048, 2048, 2048) x 4 groups."""
    shapes = [(2048, 2048, 2048)] * 4
    dtype = torch.float16

    group_a, group_b, group_c = make_group(shapes, dtype)
    # Second copy for fixed-tile run to avoid aliasing issues
    group_a2, group_b2, group_c2 = make_group(shapes, dtype)

    def fn_origami():
        tritonblas.grouped_gemm(group_a, group_b, group_c)  # BLK_M/N/K=None -> origami

    def fn_fixed():
        tritonblas.grouped_gemm(group_a2, group_b2, group_c2,
                                BLK_M=128, BLK_N=128, BLK_K=64)

    t_origami = bench_fn(fn_origami)
    t_fixed = bench_fn(fn_fixed)

    tflops_origami = compute_tflops(shapes, t_origami)
    tflops_fixed = compute_tflops(shapes, t_fixed)

    header = ["Tile Selection", "BLK_M", "BLK_N", "BLK_K", "Time (ms)", "TFLOPS"]
    rows = [
        ["origami (auto)", "auto", "auto", "auto", f"{t_origami:.3f}", f"{tflops_origami:.2f}"],
        ["fixed", "128", "128", "64", f"{t_fixed:.3f}", f"{tflops_fixed:.2f}"],
    ]

    print("\n\n=== Benchmark 4: Origami Auto-Selection vs Fixed Tiles (2048x2048x2048 x4) ===")
    print_table(header, rows)
    print(f"  Origami vs fixed speedup: {tflops_origami / tflops_fixed:.3f}x")

    assert tflops_origami > 0
    assert tflops_fixed > 0


@pytest.mark.performance
def test_benchmark5_shape_scaling():
    """N=M=K sweep in [512, 1024, 2048, 4096], group_size=4."""
    sizes = [512, 1024, 2048, 4096]
    group_size = 4
    dtype = torch.float16

    header = ["M=N=K", "group_size", "Time (ms)", "TFLOPS"]
    rows = []

    print("\n\n=== Benchmark 5: Shape Scaling (group_size=4) ===")

    for sz in sizes:
        shapes = [(sz, sz, sz)] * group_size
        group_a, group_b, group_c = make_group(shapes, dtype)

        def fn(ga=group_a, gb=group_b, gc=group_c):
            tritonblas.grouped_gemm(ga, gb, gc)

        t = bench_fn(fn)
        tflops = compute_tflops(shapes, t)
        rows.append([sz, group_size, f"{t:.3f}", f"{tflops:.2f}"])

    print_table(header, rows)

    assert len(rows) == len(sizes)
