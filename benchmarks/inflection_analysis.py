#!/usr/bin/env python3
"""Phase 1: Analyze CU sweep data to find WS vs torch inflection point."""
import json, os

def main():
    with open("results/plot_data/cu_sweep_12288_ga.json") as f:
        data = json.load(f)
    with open("results/plot_data/hierarchical_100pct_all.json") as f:
        hier_all = json.load(f)

    torch_sweep = {d["cus"]: d["tflops"] for d in data["torch"]}
    hier_sweep = {d["cus"]: d["tflops"] for d in hier_all["12288"]}
    flops = 2.0 * 12288**3

    print("=" * 70)
    print("CU Sweep Inflection: WS Hierarchical vs torch.matmul  (12K)")
    print("=" * 70)
    print()
    print(f"{'CUs':>5s}  {'torch TF':>9s}  {'WS TF':>8s}  {'delta':>7s}  {'ratio':>7s}")
    print("-" * 45)

    for cu in sorted(set(torch_sweep) & set(hier_sweep)):
        t, h = torch_sweep[cu], hier_sweep[cu]
        print(f"{cu:>5d}  {t:>9.1f}  {h:>8.1f}  {h-t:>+7.1f}  {h/t:>6.3f}x"
              f"{'  << WS wins' if h > t else ''}")

    print()
    print("=" * 70)
    print("Modeled Overlap (zero-contention CU partitioning)")
    print("=" * 70)
    print()
    comm_ms = 3.877
    torch_olap_wall = 6.767
    print(f"comm alone = {comm_ms:.3f} ms | torch overlap wall = {torch_olap_wall:.3f} ms (measured)")
    print(f"Model: wall = max(WS_GEMM_ms, comm_ms), assuming zero contention")
    print()
    print(f"{'CUs':>5s}  {'GEMM ms':>8s}  {'wall ms':>8s}  {'vs torch':>9s}")
    print("-" * 38)

    best_cu, best_wall = None, 999
    for cu in sorted(hier_sweep):
        if cu < 48: continue
        ws_ms = flops / (hier_sweep[cu] * 1e12) * 1e3
        wall = max(ws_ms, comm_ms)
        vs = wall - torch_olap_wall
        if wall < best_wall: best_wall, best_cu = wall, cu
        tag = " *** BEATS ***" if vs < 0 else ""
        print(f"{cu:>5d}  {ws_ms:>8.3f}  {wall:>8.3f}  {vs:>+9.3f}{tag}")

    print()
    print(f"Best WS: {best_cu} CUs, wall={best_wall:.3f} ms")
    pct = (torch_olap_wall - best_wall) / torch_olap_wall * 100
    print(f"vs torch overlap: {'saves' if pct > 0 else 'loses'} {abs(pct):.1f}%")

if __name__ == "__main__":
    main()
