#!/usr/bin/env python3
import time
import tritonblas
import torch
import matplotlib.pyplot as plt


def measure_heuristic_time(num_runs=100, m=512, n=512, k=512):
    """
    Measure the time taken to create a TritonBLAS heuristic and get its config.

    Args:
        num_runs (int): Number of repetitions.
        m, n, k (int): Matrix dimensions.

    Returns:
        list of float: Time in seconds for each run.
    """
    times = []
    
    x_dtype = torch.float16
    x_device = torch.device('cuda:0')

    ##
    # Initial Time
    ##
    start_time = time.perf_counter()
    selector = tritonblas.OrigamiMatmulSelector(m, n, k, x_dtype, x_dtype, x_dtype, x_device)
    end_time = time.perf_counter()
    times.append(end_time - start_time)
    # Cached Time
    for _ in range(num_runs):
        start_time = time.perf_counter()
        # Create the heuristic instance and retrieve the configuration
        config = (selector.block_m,
                  selector.block_n,
                  selector.block_k,
                  selector.group_m)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    return times


def plot_histogram(times, bins=10):
    """
    Plot a histogram of the heuristic selection times.

    Args:
        times (list of float): List of selection times.
        bins (int): Number of bins for the histogram.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(times, bins=bins, edgecolor="black")
    plt.title("Histogram of Heuristic Selection Times")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("heuristic_perf.png")


if __name__ == "__main__":
    num_runs = 100
    m, n, k = 8192, 8192, 8192  # Set matrix dimensions (adjust as necessary)
    times = measure_heuristic_time(num_runs, m, n, k)
    total_time = sum(times)
    average_time = total_time / num_runs

    print("TritonBLAS Heuristic Selection Benchmark")
    print("-----------------------------------------")
    print(f"Number of runs: {num_runs}")
    print(f"Matrix dimensions: m={m}, n={n}, k={k}")
    print(f"Initial Selection Time: {times[0]:.6f} seconds")
    print(f"Total time over {num_runs} runs: {total_time:.6f} seconds")
    print(f"Average time per heuristic selection: {average_time:.6f} seconds")
