"""
The gate array's layout is the *transport's*, not the kernel's, and the two in use disagree.

The layouts under test are plain int32 counters at unit stride and 64-byte gate structs whose ready
word is the first int64, i.e. a stride of 8 elements. A strided tensor view cannot express that
difference, because Triton indexes the raw pointer and ignores torch strides -- so the stride is a
kernel parameter, and this checks it.

Deliberately transport-free: a new transport's layout can be validated here before any collective is
wired up, and a failure points at the epilogue rather than at somebody's fabric.
"""

import pytest
import torch

from tritonblas.config import matmul_preamble
from tritonblas.matmul import matmul_lt
from tritonblas.origami import OrigamiMatmulSelector
from tritonblas.schedule import schedule

DEVICE = "cuda"


def _band_signal_table(tiles_m, tiles_n, tiles_per_signal, device):
    """One signal per band of `tiles_per_signal` tile-rows, the grouping a row-band collective uses."""
    table = torch.empty(tiles_m * tiles_n, dtype=torch.int32, device=device)
    for tm in range(tiles_m):
        for tn in range(tiles_n):
            table[tm * tiles_n + tn] = tm // tiles_per_signal
    return table


@pytest.mark.parametrize(
    "gate_stride, gate_dtype, gate_base",
    [
        (1, torch.int32, 0),   # plain counters
        (8, torch.int64, 0),   # 64-byte gate struct, ready word first
        (8, torch.int64, 3),   # a rank's offset need not be a multiple of the stride
    ],
)
def test_gate_layout_signals_exactly_the_right_words(gate_stride, gate_dtype, gate_base):
    """Every gate takes one increment per producing tile, and nothing else is written."""
    m = n = k = 1024
    a = torch.randn(m, k, dtype=torch.float16, device=DEVICE)
    b = torch.randn(k, n, dtype=torch.float16, device=DEVICE)
    c = torch.empty(m, n, dtype=torch.float16, device=DEVICE)

    s = schedule(a, b)
    tiles_per_signal = 2
    assert s.tiles_m % tiles_per_signal == 0
    num_signals = s.tiles_m // tiles_per_signal
    table = _band_signal_table(s.tiles_m, s.tiles_n, tiles_per_signal, DEVICE)

    selector = OrigamiMatmulSelector(m, n, k, torch.float16, torch.float16, torch.float16,
                                     torch.device(DEVICE, 0))
    config = matmul_preamble(selector)

    # Padded past the last gate so a stride or base that overruns is caught rather than landing
    # outside the allocation.
    gates = torch.zeros(gate_base + num_signals * gate_stride + 5,
                        dtype=gate_dtype, device=DEVICE)
    matmul_lt(a, b, c, selector, config, gates=gates, signal_of_tile=table,
              gate_stride=gate_stride, gate_base=gate_base)
    torch.cuda.synchronize()

    got = gates.cpu()
    expected = tiles_per_signal * s.tiles_n
    for g in range(num_signals):
        assert got[gate_base + g * gate_stride].item() == expected, \
            f"gate {g}: {got[gate_base + g * gate_stride].item()} increments, expected {expected}"

    touched = {gate_base + g * gate_stride for g in range(num_signals)}
    spilled = [i for i in range(got.numel()) if i not in touched and got[i].item() != 0]
    assert not spilled, f"words written that are not gates: {spilled}"
