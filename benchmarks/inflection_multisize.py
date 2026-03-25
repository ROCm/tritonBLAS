#!/usr/bin/env python3
"""Analyze wave quantization impact across problem sizes, find which sizes
have the most favorable inflection for WS vs torch at 304 CUs."""
import json

sizes = [2048, 4096, 8192, 12288, 16384]
BLK = 256

print(f"{'Size':>6s}  {'tiles':>6s}  {'waves':>6s}  {'last_wave':>10s}  {'occ%':>6s}  {'WQ penalty':>11s}")
print("-" * 60)

for sz in sizes:
    tiles_per_dim = sz // BLK
    total_tiles = tiles_per_dim ** 2
    full_waves = total_tiles // 304
    last_wave_tiles = total_tiles - full_waves * 304
    if last_wave_tiles == 0:
        occupancy = 100.0
        penalty = "none"
    else:
        occupancy = last_wave_tiles / 304 * 100
        penalty = f"{(1 - occupancy/100) * (1/(full_waves+1)) * 100:.1f}%"
    print(f"{sz:>6d}  {total_tiles:>6d}  {full_waves+1 if last_wave_tiles else full_waves:>6d}  "
          f"{last_wave_tiles:>10d}  {occupancy:>5.1f}%  {penalty:>11s}")

print()
print("Sizes with worst wave quantization at 304 CUs have largest WS advantage")
print()

for sz in sizes:
    fname_ga = f"results/plot_data/cu_sweep_{sz}_ga.json"
    fname_hier = "results/plot_data/hierarchical_100pct_all.json"
    try:
        with open(fname_ga) as f:
            data = json.load(f)
        with open(fname_hier) as f:
            hier_all = json.load(f)

        torch_at_304 = next((d["tflops"] for d in data["torch"] if d["cus"] == 304), None)
        hier_at_304 = next((d["tflops"] for d in hier_all[str(sz)] if d["cus"] == 304), None)
        
        if torch_at_304 and hier_at_304:
            gap = (hier_at_304 - torch_at_304) / torch_at_304 * 100
            print(f"  {sz:>6d}  torch@304={torch_at_304:>7.1f} TF  WS@304={hier_at_304:>7.1f} TF  gap={gap:>+6.1f}%")
    except Exception as e:
        print(f"  {sz:>6d}  data not available: {e}")

print()
print("=== Per-size CU sweep comparison (selected CU counts) ===")
for sz in sizes:
    try:
        with open(f"results/plot_data/cu_sweep_{sz}_ga.json") as f:
            data = json.load(f)
        with open("results/plot_data/hierarchical_100pct_all.json") as f:
            hier_all = json.load(f)
        
        torch_sweep = {d["cus"]: d["tflops"] for d in data["torch"]}
        hier_sweep = {d["cus"]: d["tflops"] for d in hier_all[str(sz)]}
        
        print(f"\n  {sz}x{sz}:")
        for cu in [256, 272, 288, 304]:
            t = torch_sweep.get(cu, 0)
            h = hier_sweep.get(cu, 0)
            if t > 0 and h > 0:
                flops = 2.0 * sz**3
                t_ms = flops / (t * 1e12) * 1e3
                h_ms = flops / (h * 1e12) * 1e3
                print(f"    CU={cu:>3d}  torch={t:>7.1f} TF ({t_ms:.3f} ms)  WS={h:>7.1f} TF ({h_ms:.3f} ms)  WS/torch={h/t:.3f}x")
    except Exception:
        pass
