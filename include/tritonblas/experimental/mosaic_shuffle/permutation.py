"""
Feistel network permutation for bijective shuffling.

The Feistel network with cycle-walking requires branching that Triton's
JIT compiler cannot handle. Instead, we compute the full permutation
table on the host (CPU) using the pure-Python reference, then pass it
to kernels as a lookup tensor. The kernel does a single tl.load -- zero
branching, zero type issues.
"""

import math
import random
from typing import Tuple

import torch


def _feistel_round_ref(val: int, key: int) -> int:
    """Python reference of the Feistel round function."""
    x = ((val ^ key) * 0x9E3779B9) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFFFFFFFFFF
    return x


def feistel_permute_ref(index: int, n: int, half_bits: int, mask: int,
                        key0: int, key1: int, key2: int, key3: int) -> int:
    """Python reference of the Feistel permutation with cycle-walking."""
    x = index
    for _ in range(8):
        L = x >> half_bits
        R = x & mask
        L, R = R, L ^ (_feistel_round_ref(R, key0) & mask)
        L, R = R, L ^ (_feistel_round_ref(R, key1) & mask)
        L, R = R, L ^ (_feistel_round_ref(R, key2) & mask)
        L, R = R, L ^ (_feistel_round_ref(R, key3) & mask)
        x = (L << half_bits) | R
        if x < n:
            return x
    return index


def compute_feistel_params(
    n: int,
    seed: int = 42,
) -> Tuple[int, int, int, int, int, int]:
    """
    Compute Feistel network parameters for a domain of size n.

    Returns:
        (half_bits, mask, key0, key1, key2, key3)
    """
    if n < 2:
        raise ValueError(f"Feistel permutation requires n >= 2, got {n}")

    bits = max(2, math.ceil(math.log2(n)))
    if bits % 2 == 1:
        bits += 1
    half_bits = bits // 2
    mask = (1 << half_bits) - 1

    rng = random.Random(seed)
    keys = [rng.getrandbits(32) for _ in range(4)]

    return half_bits, mask, keys[0], keys[1], keys[2], keys[3]


def compute_permutation_table(n: int, seed: int = 42, device=None) -> torch.Tensor:
    """
    Compute a full bijective permutation table [0, n) -> [0, n) on the host,
    then move it to the specified device.

    Uses a Feistel network with cycle-walking (computed in pure Python on CPU).
    The resulting tensor can be passed to a Triton kernel for branchless lookup
    via tl.load(perm_table + index).

    Returns:
        torch.Tensor of shape (n,) with dtype int32 on the target device.
    """
    if n < 2:
        raise ValueError(f"Permutation table requires n >= 2, got {n}")

    half_bits, mask, k0, k1, k2, k3 = compute_feistel_params(n, seed)
    table = [feistel_permute_ref(i, n, half_bits, mask, k0, k1, k2, k3) for i in range(n)]

    t = torch.tensor(table, dtype=torch.int32)
    if device is not None:
        t = t.to(device)
    return t
