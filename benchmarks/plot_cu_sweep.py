#!/usr/bin/env python3
"""
Plot CU sweep results for matrix multiplication benchmarks.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_cu_sweep(csv_file, output_file=None, title=None):
    """
    Create a CU sweep plot from benchmark CSV data.
    
    Args:
        csv_file: Path to CSV file with CU sweep results
        output_file: Path to save the plot (if None, displays instead)
        title: Custom title for the plot
    """
    # Read the CSV
    df = pd.read_csv(csv_file)
    
    # Extract matrix size from first row
    m, n, k = df.iloc[0]['m'], df.iloc[0]['n'], df.iloc[0]['k']
    in_dtype = df.iloc[0]['in_dtype']
    out_dtype = df.iloc[0]['out_dtype']
    
    # Default title if not provided
    if title is None:
        title = f"{m}x{n}x{k} {in_dtype.upper()} GEMM — CU Sweep (MI300X)"
    
    # Group by mode and active_cus
    modes = df['mode'].unique()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color and style mapping
    mode_styles = {
        'persistent': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'label': 'Persistent'},
        'streamk': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'label': 'Stream-K'},
        'work_stealing': {'color': '#2ca02c', 'linestyle': '--', 'marker': '^', 'label': 'Work-Stealing'},
    }
    
    # Plot each mode
    for mode in sorted(modes):
        mode_data = df[df['mode'] == mode].sort_values('active_cus')
        
        style = mode_styles.get(mode, {'color': 'gray', 'linestyle': '-', 'marker': 'o', 'label': mode})
        
        ax.plot(
            mode_data['active_cus'],
            mode_data['tritonblas_gflops'],
            color=style['color'],
            linestyle=style['linestyle'],
            marker=style['marker'],
            markersize=6,
            linewidth=2,
            label=style['label']
        )
    
    # Check if there's torch data to add
    # (In case you want to add torch.mm reference later)
    
    # Formatting
    ax.set_xlabel('Active CUs', fontsize=12, fontweight='bold')
    ax.set_ylabel('GFLOPS', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Y-axis formatting
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CU sweep benchmark results")
    parser.add_argument("csv_file", help="Path to CSV file with CU sweep results")
    parser.add_argument("--output", "-o", help="Output file path (default: show plot)")
    parser.add_argument("--title", "-t", help="Custom plot title")
    
    args = parser.parse_args()
    
    plot_cu_sweep(args.csv_file, args.output, args.title)
