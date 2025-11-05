#!/usr/bin/env python3
import yaml
import argparse
import torch


def get_all_dtypes():
    """Get all available PyTorch dtypes for benchmarking"""
    standard_dtypes = ["float32", "float16", "bfloat16"]
    fp8_dtypes = ["float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2"]
    int_dtypes = ["int8"]
    
    # Check which dtypes are actually available
    available = []
    for dtype_name in standard_dtypes + fp8_dtypes + int_dtypes:
        if hasattr(torch, dtype_name):
            available.append(dtype_name)
    
    return available


def main():
    available_dtypes = get_all_dtypes()
    
    parser = argparse.ArgumentParser(
        description="Generate a YAML file with benchmark sizes for matmul using dimensions from 128 to 8192 in steps of 128, including transpose options and data types."
    )
    parser.add_argument(
        "--transA",
        type=str,
        choices=["N", "T"],
        default="T",
        help="Transpose type for A matrix ('N' for no transpose, 'T' for transpose, default: 'T').",
    )
    parser.add_argument(
        "--transB",
        type=str,
        choices=["N", "T"],
        default="N",
        help="Transpose type for B matrix ('N' for no transpose, 'T' for transpose, default: 'N').",
    )
    parser.add_argument(
        "--in-dtype",
        type=str,
        choices=available_dtypes,
        default="float16",
        help=f"Input data type. Available: {', '.join(available_dtypes)} (default: 'float16').",
    )
    parser.add_argument(
        "--out-dtype",
        type=str,
        choices=available_dtypes,
        default="float16",
        help=f"Output data type. Available: {', '.join(available_dtypes)} (default: 'float16').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output YAML filename (default: auto-generated based on parameters).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=128,
        help="Minimum dimension size (default: 128).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=8192,
        help="Maximum dimension size (default: 8192).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=128,
        help="Step size for dimensions (default: 128).",
    )
    args = parser.parse_args()

    # Validate dtype
    if args.in_dtype not in available_dtypes:
        print(f"Warning: {args.in_dtype} may not be available. Available dtypes: {available_dtypes}")
    if args.out_dtype not in available_dtypes:
        print(f"Warning: {args.out_dtype} may not be available. Available dtypes: {available_dtypes}")

    # Create output filename
    if args.output:
        output_filename = args.output
    else:
        output_filename = (
            f"matmul_grid_{args.transA}{args.transB}_{args.in_dtype}_{args.out_dtype}.yaml"
        )

    # Generate dataset based on the transpose and data type options
    entries = []
    for m in range(args.min_size, args.max_size + 1, args.step):
        for n in range(args.min_size, args.max_size + 1, args.step):
            for k in range(args.min_size, args.max_size + 1, args.step):
                entries.append(
                    {
                        "in_dtype": f"torch.{args.in_dtype}",
                        "out_dtype": f"torch.{args.out_dtype}",
                        "transA": str(args.transA),
                        "transB": str(args.transB),
                        "m": m,
                        "n": n,
                        "k": k,
                    }
                )

    # Save the dataset to the output YAML file
    with open(output_filename, "w") as f:
        yaml.dump(entries, f, default_flow_style=False, sort_keys=False)

    print(f"Generated {len(entries)} entries and saved to '{output_filename}'.")
    print(f"  Dimensions: {args.min_size} to {args.max_size} (step: {args.step})")
    print(f"  Transpose: A={args.transA}, B={args.transB}")
    print(f"  Dtypes: in={args.in_dtype}, out={args.out_dtype}")


if __name__ == "__main__":
    main()
