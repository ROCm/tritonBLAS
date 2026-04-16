import json
import argparse
import torch
import tritonblas  # Ensure this module is available in your environment


def load_input_configs(filename):
    with open(filename, "r") as f:
        return json.load(f)


def process_configs(configs):
    results = []
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OrigamiMatmulSelector")
    device = torch.device("cuda:0")
    # Default dtypes - can be made configurable if needed
    a_dtype = torch.float16
    b_dtype = torch.float16
    c_dtype = torch.float16

    for cfg in configs:
        m, n, k = cfg["m"], cfg["n"], cfg["k"]
        selector = tritonblas.OrigamiMatmulSelector(m, n, k, a_dtype, b_dtype, c_dtype, device)
        BLK_M = selector.block_m
        BLK_N = selector.block_n
        BLK_K = selector.block_k
        gsize_m = selector.group_m
        results.append({"m": m, "n": n, "k": k, "BLK_M": BLK_M, "BLK_N": BLK_N, "BLK_K": BLK_K, "gsize_m": gsize_m})
    return results


def save_output_configs(results, filename):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="TritonBLAS Config Exporter")
    parser.add_argument("input_json", help="Input JSON file containing matmul m,n,k sizes.")
    parser.add_argument("output_json", help="Output JSON file to write the config.")
    args = parser.parse_args()

    configs = load_input_configs(args.input_json)
    results = process_configs(configs)
    save_output_configs(results, args.output_json)


if __name__ == "__main__":
    main()
