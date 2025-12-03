#!/usr/bin/env python3
"""
FP4 Dataset Generator

Generate YAML benchmark datasets for FP4 matrix multiplication.
Ensures K dimensions are divisible by 32 (FP4 requirement).
"""
import yaml
import argparse


# Production-realistic problem sizes from benchmark_fp4.py
PRODUCTION_SIZES = [
    # Pure compute
    (256, 2048, 8192),
    (2048, 8192, 8192),
    (16384, 16384, 16384),
    # QKV projection - various batch sizes
    (1, 1280, 8192),
    (64, 1280, 8192),
    (128, 1280, 8192),
    (256, 1280, 8192),
    (512, 1280, 8192),
    (1024, 1280, 8192),
    (2048, 1280, 8192),
    (4096, 1280, 8192),
    # Attention output - various batch sizes
    (1, 8192, 1024),
    (64, 8192, 1024),
    (128, 8192, 1024),
    (256, 8192, 1024),
    (512, 8192, 1024),
    (1024, 8192, 1024),
    (2048, 8192, 1024),
    (4096, 8192, 1024),
]


def validate_k(k):
    """Check if K is divisible by 32 (FP4 requirement)"""
    return k % 32 == 0


def round_k_to_valid(k):
    """Round K to nearest valid value (divisible by 32)"""
    return ((k + 15) // 32) * 32


def generate_grid_dataset(min_size, max_size, step, transA, transB, out_dtype, validate_k_flag=True):
    """Generate a grid of problem sizes with K validation"""
    entries = []
    
    for m in range(min_size, max_size + 1, step):
        for n in range(min_size, max_size + 1, step):
            for k in range(min_size, max_size + 1, step):
                # Ensure K is divisible by 32
                if validate_k_flag and not validate_k(k):
                    k_valid = round_k_to_valid(k)
                    if k_valid > max_size:
                        continue
                    k = k_valid
                
                entries.append({
                    "in_dtype": "fp4",
                    "out_dtype": f"torch.{out_dtype}",
                    "transA": str(transA),
                    "transB": str(transB),
                    "m": m,
                    "n": n,
                    "k": k,
                })
    
    # Remove duplicates (can occur due to K rounding)
    seen = set()
    unique_entries = []
    for entry in entries:
        key = (entry["m"], entry["n"], entry["k"], entry["transA"], entry["transB"])
        if key not in seen:
            seen.add(key)
            unique_entries.append(entry)
    
    return unique_entries


def generate_production_dataset(transA, transB, out_dtype):
    """Generate production-realistic problem sizes"""
    entries = []
    
    for m, n, k in PRODUCTION_SIZES:
        entries.append({
            "in_dtype": "fp4",
            "out_dtype": f"torch.{out_dtype}",
            "transA": str(transA),
            "transB": str(transB),
            "m": m,
            "n": n,
            "k": k,
        })
    
    return entries


def generate_small_dataset(transA, transB, out_dtype):
    """Generate small test dataset (128-1024 range)"""
    return generate_grid_dataset(128, 1024, 128, transA, transB, out_dtype)


def generate_large_dataset(transA, transB, out_dtype):
    """Generate large-scale dataset (4096-16384 range)"""
    return generate_grid_dataset(4096, 16384, 1024, transA, transB, out_dtype)


def main():
    parser = argparse.ArgumentParser(
        description="Generate YAML benchmark datasets for FP4 matrix multiplication. "
                    "Ensures K dimensions are divisible by 32 (FP4 requirement)."
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["grid", "production", "small", "large"],
        default="grid",
        help="Dataset preset: 'grid' (custom grid), 'production' (LLM workloads), "
             "'small' (128-1024), 'large' (4096-16384). Default: 'grid'.",
    )
    parser.add_argument(
        "--transA",
        type=str,
        choices=["N", "T"],
        default="T",
        help="Transpose type for A matrix ('N' for no transpose, 'T' for transpose). Default: 'T'.",
    )
    parser.add_argument(
        "--transB",
        type=str,
        choices=["N", "T"],
        default="N",
        help="Transpose type for B matrix ('N' for no transpose, 'T' for transpose). Default: 'N'.",
    )
    parser.add_argument(
        "--out-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Output data type. Default: 'bfloat16'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output YAML filename. Default: auto-generated based on preset.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=128,
        help="Minimum dimension size for 'grid' preset. Default: 128.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=8192,
        help="Maximum dimension size for 'grid' preset. Default: 8192.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=256,
        help="Step size for dimensions in 'grid' preset. Default: 256.",
    )
    parser.add_argument(
        "--no-validate-k",
        action="store_true",
        help="Disable K divisibility validation (not recommended for FP4).",
    )
    
    args = parser.parse_args()
    
    # Generate dataset based on preset
    if args.preset == "production":
        entries = generate_production_dataset(args.transA, args.transB, args.out_dtype)
        default_output = f"matmul_fp4_production_{args.transA}{args.transB}_{args.out_dtype}.yaml"
    elif args.preset == "small":
        entries = generate_small_dataset(args.transA, args.transB, args.out_dtype)
        default_output = f"matmul_fp4_small_{args.transA}{args.transB}_{args.out_dtype}.yaml"
    elif args.preset == "large":
        entries = generate_large_dataset(args.transA, args.transB, args.out_dtype)
        default_output = f"matmul_fp4_large_{args.transA}{args.transB}_{args.out_dtype}.yaml"
    else:  # grid
        entries = generate_grid_dataset(
            args.min_size, args.max_size, args.step,
            args.transA, args.transB, args.out_dtype,
            validate_k_flag=not args.no_validate_k
        )
        default_output = f"matmul_fp4_grid_{args.transA}{args.transB}_{args.out_dtype}.yaml"
    
    # Determine output filename
    output_filename = args.output if args.output else default_output
    
    # Save dataset to YAML file
    with open(output_filename, "w") as f:
        yaml.dump(entries, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated {len(entries)} FP4 benchmark entries and saved to '{output_filename}'.")
    print(f"  Preset: {args.preset}")
    if args.preset == "grid":
        print(f"  Dimensions: {args.min_size} to {args.max_size} (step: {args.step})")
    print(f"  Transpose: A={args.transA}, B={args.transB}")
    print(f"  Dtypes: in=fp4, out={args.out_dtype}")
    print(f"  K validation: {'disabled' if args.no_validate_k else 'enabled (K % 32 == 0)'}")


if __name__ == "__main__":
    main()
