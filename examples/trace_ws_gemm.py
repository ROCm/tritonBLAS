import torch
import tritonblas
import argparse


def trace_gemm(m, n, k, output):
    # Allocate Tensors
    A = torch.randn(m, k, device="cuda", dtype=torch.float16)
    B = torch.randn(n, k, device="cuda", dtype=torch.float16).T
    C = torch.zeros((m, n), device="cuda", dtype=torch.float16)

    # Query Origami for tile config
    selector = tritonblas.OrigamiMatmulSelector(
        m, n, k, A.dtype, B.dtype, C.dtype, A.device
    )
    cfg = tritonblas.matmul_preamble(selector)
    cfg.global_atomic = False

    # Warmup (compile the kernel without tracing overhead)
    tritonblas.matmul_lt(A, B, C, selector, cfg)
    torch.cuda.synchronize()

    # Run with tracing enabled
    C, trace_data = tritonblas.matmul_lt(A, B, C, selector, cfg, enable_streamk=False, work_stealing=True, trace=True)

    # Render the per-pid tile timeline to PNG
    tritonblas.plot_gemm_trace(trace_data, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trace persistent GEMM tile execution and output a timeline PNG."
    )
    parser.add_argument("--m", type=int, default=8192, help="Rows of A / C (default: 8192)")
    parser.add_argument("--n", type=int, default=8192, help="Columns of B / C (default: 8192)")
    parser.add_argument("--k", type=int, default=8192, help="Inner dimension (default: 8192)")
    parser.add_argument("-o", "--output", type=str, default="ws_gemm_trace.png",
                        help="Output PNG path (default: ws_gemm_trace.png)")
    args = parser.parse_args()
    trace_gemm(args.m, args.n, args.k, args.output)
