import torch
import triton
import tritonblas
import argparse


def example_matmul(m, n, k):
    # Allocate Tensors
    A = torch.randn(m, k, device="cuda", dtype=torch.float16)
    B = torch.randn(n, k, device="cuda", dtype=torch.float16).T
    C = torch.zeros((m, n), device="cuda", dtype=torch.float16)

    Adesc = True
    Bdesc = False

    # Find matmul config
    # TODO: Add a config class to tritonblas.

    # Run TritonBLAS matmul
    tritonblas.matmul(A, B, C, Adesc, Bdesc)

    # Print result
    print(C)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example TritonBLAS matrix multiplication with CLI parameters for m, n, k."
    )
    parser.add_argument(
        "--m",
        type=int,
        default=8192,
        help="Number of rows in matrix A and C (default: 8192)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8192,
        help="Number of columns in matrix B (after transpose) and C (default: 8192)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8192,
        help="Number of columns in matrix A and rows in matrix B (default: 8192)",
    )
    args = parser.parse_args()
    example_matmul(args.m, args.n, args.k)
