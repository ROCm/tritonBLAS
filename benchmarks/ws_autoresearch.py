#!/usr/bin/env python3
"""Scalable autoresearch runner for WS kernel optimization.

Runs experiments in parallel across GPUs. Each experiment modifies kernel
parameters and measures performance.

Usage:
    python3 benchmarks/ws_autoresearch.py --wave <wave_name>
"""
import torch
import tritonblas
from tritonblas.kernels import ws_persistent_matmul
from tritonblas.config import COUNTER_STRIDE, matmul_preamble
import statistics
import json
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

RESULTS_DIR = Path("results/ws_autoresearch")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SIZES = [4096, 8192, 16384]
PRIMARY_SIZE = 8192


def run_experiment_on_gpu(gpu_id, exp_id, config):
    """Run a single experiment on a specific GPU. Called in a subprocess."""
    import torch
    import tritonblas
    from tritonblas.kernels import ws_persistent_matmul
    from tritonblas.config import COUNTER_STRIDE, matmul_preamble
    import statistics

    torch.cuda.set_device(gpu_id)
    dev = torch.device("cuda", gpu_id)
    stream = torch.cuda.Stream(device=dev)

    results = {}
    for sz in config.get("sizes", [PRIMARY_SIZE]):
        M = N = K = sz
        FLOPS = 2.0 * M * N * K
        A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        B = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
        C = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        ref = torch.matmul(A.float(), B.float()).bfloat16()

        # torch baseline
        for _ in range(10):
            torch.matmul(A, B, out=C)
        torch.cuda.synchronize()
        t = []
        for _ in range(30):
            torch.cuda.synchronize()
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            torch.matmul(A, B, out=C)
            en.record()
            torch.cuda.synchronize()
            t.append(st.elapsed_time(en))
        torch_ms = statistics.median(t)

        # WS with experiment config
        sel = tritonblas.OrigamiMatmulSelector(M, N, K, A.dtype, B.dtype, C.dtype, dev)
        if "cpx" in config:
            sel.COUNTERS_PER_XCD = config["cpx"]
        if "gm" in config:
            sel._workgroup_mapping = config["gm"]
        cfg = matmul_preamble(sel)

        grids = config.get("grids", sel._hardware.N_CU)
        num_xcds = sel.num_sms
        BM = config.get("block_m", sel.block_m)
        BN = config.get("block_n", sel.block_n)
        BK = config.get("block_k", sel.block_k)
        ns = config.get("num_stages", 2)
        nw = config.get("num_warps", 8)
        wpe = config.get("waves_per_eu", 0)
        gm = config.get("gm", sel.group_m)
        cpx = config.get("cpx", sel.COUNTERS_PER_XCD)
        ga = config.get("global_atomic", False)

        mask = cfg.mask
        if grids > len(mask):
            mask = torch.ones(grids, dtype=torch.int32, device=dev)

        try:
            def launch():
                ws_persistent_matmul[(grids,)](
                    A, B, C, None, None, None, cfg.tile_counter,
                    M, N, K, A.stride(0), B.stride(1), C.stride(0), C.stride(1), 0,
                    stride_ak=A.stride(1), stride_bk=B.stride(0),
                    BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
                    GROUP_SIZE_M=gm, NUM_SMS=grids, NUM_XCDS=num_xcds,
                    COUNTERS_PER_XCD=cpx, COUNTER_STRIDE=COUNTER_STRIDE,
                    BIAS=False, EVEN_K=(K % BK == 0),
                    CACHE_MODIFIER_A=config.get("cache_a"),
                    CACHE_MODIFIER_B=config.get("cache_b"),
                    QUANTIZED=False, GLOBAL_ATOMIC=ga,
                    num_stages=ns, num_warps=nw, waves_per_eu=wpe,
                    matrix_instr_nonkdim=config.get("mfma", 16),
                    kpack=config.get("kpack", 1),
                    mask_ptr=mask,
                )

            def rst():
                cfg.reset(work_stealing=True)

            for _ in range(10):
                with torch.cuda.stream(stream):
                    rst()
                    launch()
            torch.cuda.synchronize()

            # Correctness
            rst()
            launch()
            torch.cuda.synchronize()
            err = (C.float() - ref.float()).abs().max().item()
            if err > 10:
                results[sz] = {"status": "INCORRECT", "err": err}
                continue

            t = []
            for _ in range(50):
                with torch.cuda.stream(stream):
                    rst()
                torch.cuda.synchronize()
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                st.record(stream)
                with torch.cuda.stream(stream):
                    launch()
                en.record(stream)
                torch.cuda.synchronize()
                t.append(st.elapsed_time(en))

            ws_ms = statistics.median(t)
            ws_tf = FLOPS / (ws_ms * 1e-3) / 1e12
            to_tf = FLOPS / (torch_ms * 1e-3) / 1e12
            gap = (ws_ms / torch_ms - 1) * 100

            results[sz] = {
                "status": "OK",
                "ws_ms": ws_ms,
                "torch_ms": torch_ms,
                "ws_tflops": ws_tf,
                "torch_tflops": to_tf,
                "gap_pct": gap,
            }
        except Exception as e:
            results[sz] = {"status": "FAILED", "error": str(e)[:200]}

        del A, B, C, ref
        torch.cuda.empty_cache()

    return {"exp_id": exp_id, "gpu": gpu_id, "config": config, "results": results}


def run_wave(wave_name, experiments):
    """Run a wave of experiments in parallel across GPUs."""
    n_gpus = torch.cuda.device_count()
    print(f"\n{'='*70}")
    print(f"  WAVE: {wave_name} ({len(experiments)} experiments, {n_gpus} GPUs)")
    print(f"  {datetime.now().isoformat()}")
    print(f"{'='*70}")

    all_results = []

    for batch_start in range(0, len(experiments), n_gpus):
        batch = experiments[batch_start:batch_start + n_gpus]
        procs = []

        for i, (exp_id, config) in enumerate(batch):
            gpu_id = i % n_gpus
            config["sizes"] = [PRIMARY_SIZE]
            print(f"  [{exp_id}] GPU {gpu_id}: {config.get('desc', exp_id)}")

            # Run in subprocess for GPU isolation
            proc = subprocess.Popen(
                [sys.executable, "-c", f"""
import json, sys
sys.path.insert(0, '.')
from benchmarks.ws_autoresearch import run_experiment_on_gpu
result = run_experiment_on_gpu({gpu_id}, '{exp_id}', {json.dumps(config)})
print(json.dumps(result))
"""],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=120,
            )
            procs.append((exp_id, proc))

        for exp_id, proc in procs:
            try:
                stdout, stderr = proc.communicate(timeout=120)
                if proc.returncode == 0 and stdout.strip():
                    for line in stdout.strip().split("\n"):
                        try:
                            result = json.loads(line)
                            all_results.append(result)
                            r = result["results"].get(PRIMARY_SIZE, {})
                            if r.get("status") == "OK":
                                beat = " *** BEATS" if r["gap_pct"] < 0 else ""
                                print(f"    {exp_id}: {r['ws_ms']:.3f}ms "
                                      f"({r['ws_tflops']:.0f}TF) gap={r['gap_pct']:+.1f}%{beat}")
                            else:
                                print(f"    {exp_id}: {r.get('status', 'UNKNOWN')}")
                            break
                        except json.JSONDecodeError:
                            continue
                else:
                    print(f"    {exp_id}: FAILED (rc={proc.returncode})")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"    {exp_id}: TIMEOUT")

    # Rank results
    valid = [(r["exp_id"], r["results"].get(PRIMARY_SIZE, {}))
             for r in all_results if r["results"].get(PRIMARY_SIZE, {}).get("status") == "OK"]
    valid.sort(key=lambda x: x[1]["ws_ms"])

    print(f"\n  WAVE {wave_name} RESULTS (ranked):")
    for i, (eid, r) in enumerate(valid):
        beat = " <<< BEATS TORCH" if r["gap_pct"] < 0 else ""
        print(f"    {i+1}. {eid:<30} {r['ws_ms']:.3f}ms {r['ws_tflops']:.0f}TF "
              f"({r['gap_pct']:+.1f}%){beat}")

    # Save wave results
    wave_file = RESULTS_DIR / f"wave_{wave_name}_{datetime.now().strftime('%H%M%S')}.json"
    with open(wave_file, "w") as f:
        json.dump({"wave": wave_name, "timestamp": datetime.now().isoformat(),
                    "results": all_results}, f, indent=2)
    print(f"  Saved to {wave_file}")

    return all_results


if __name__ == "__main__":
    wave = sys.argv[1] if len(sys.argv) > 1 else "test"

    if wave == "test":
        # Quick sanity test
        exps = [
            ("baseline", {"desc": "current defaults"}),
            ("wpe1", {"desc": "waves_per_eu=1", "waves_per_eu": 1}),
        ]
        run_wave("test", exps)
