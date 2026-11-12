"""
Validate the gate-layout parameters with no transport and no pytest fixtures.

Same assertions as `tests/test_schedule.py::test_gate_layout_signals_exactly_the_right_words`, run
standalone because the branch's conftest does not import against the origami build in this
environment. Covers both layouts in use: unit-stride int32 counters, and 64-byte structs whose ready
word is the first int64 (stride 8).
"""
import torch, tritonblas
from tritonblas.config import matmul_preamble
from tritonblas.matmul import matmul_lt
from tritonblas.origami import OrigamiMatmulSelector
from tritonblas.schedule import schedule

DEV = "cuda"
m = n = k = 1024
a = torch.randn(m, k, dtype=torch.float16, device=DEV)
b = torch.randn(k, n, dtype=torch.float16, device=DEV)
c = torch.empty(m, n, dtype=torch.float16, device=DEV)
s = schedule(a, b)
tps = 2
num_signals = s.tiles_m // tps
table = torch.empty(s.tiles_m * s.tiles_n, dtype=torch.int32, device=DEV)
for tm in range(s.tiles_m):
    for tn in range(s.tiles_n):
        table[tm * s.tiles_n + tn] = tm // tps

sel = OrigamiMatmulSelector(m, n, k, torch.float16, torch.float16, torch.float16, torch.device(DEV, 0))
cfg = matmul_preamble(sel)
ok = True
for stride, dtype, base in ((1, torch.int32, 0), (8, torch.int64, 0), (8, torch.int64, 3)):
    span = base + num_signals * stride
    gates = torch.zeros(span + 5, dtype=dtype, device=DEV)
    matmul_lt(a, b, c, sel, cfg, gates=gates, signal_of_tile=table,
              gate_stride=stride, gate_base=base)
    torch.cuda.synchronize()
    got = gates.cpu()
    want = tps * s.tiles_n
    touched = {base + g * stride for g in range(num_signals)}
    bad = [g for g in range(num_signals) if got[base + g * stride].item() != want]
    spill = [i for i in range(got.numel()) if i not in touched and got[i].item() != 0]
    status = "ok" if not bad and not spill else "FAIL"
    ok &= not bad and not spill
    print(f"  stride={stride} dtype={str(dtype).split('.')[-1]:7s} base={base}: {status} "
          f"(gates={num_signals}, expected {want} each, wrong={len(bad)}, spilled={len(spill)})")
print("PASS" if ok else "FAIL")
