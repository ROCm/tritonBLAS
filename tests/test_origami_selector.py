# SPDX-License-Identifier: MIT
"""Regression tests: OrigamiMatmulSelector.num_sms must never be 0.

rocm-origami reports "no XCD remapping" as wgmxcc == 0, where older origami
returned 1. num_sms is passed to the kernels as NUM_XCDS (1 = no remapping)
and matmul_fp4 divides by it, so a 0 leaking out crashed small FP4 problems
with ZeroDivisionError.
"""

import pytest
import torch
import origami

from tritonblas.origami import OrigamiMatmulSelector
from conftest import requires_fp4

DEVICE = torch.device("cuda", torch.cuda.current_device())


def _selector(m, n, k, dtype, streamk=False):
    if dtype == "f4":
        return OrigamiMatmulSelector(m, n, k, "f4", "f4", torch.bfloat16, DEVICE, mx_block_size=32)
    return OrigamiMatmulSelector(m, n, k, dtype, dtype, dtype, DEVICE, streamk=streamk)


def test_no_remap_from_origami_is_reported_as_one_xcd(monkeypatch):
    """Force origami's "no remapping" value so this runs the same on any GPU."""
    real_select = origami.select_workgroup_mapping

    def select_no_remap(*args):
        mapping = real_select(*args)
        mapping.wgmxcc = 0
        return mapping

    monkeypatch.setattr(origami, "select_workgroup_mapping", select_no_remap)
    assert _selector(4096, 4096, 4096, torch.bfloat16).num_sms == 1


# Small problems for which rocm-origami 0.0.3 returns wgmxcc == 0 on MI355X.
@pytest.mark.parametrize("streamk", [False, True])
@pytest.mark.parametrize("m, n, k", [(32, 32, 32), (16, 16, 4096)])
def test_small_problems_report_at_least_one_xcd(m, n, k, streamk):
    assert _selector(m, n, k, torch.bfloat16, streamk).num_sms >= 1


# (128, 64, 64) and (256, 128, 128) are the selector inputs matmul_fp4 uses for
# 128^3 and 256^3 problems (it passes the packed operand shapes); both crashed.
@requires_fp4
@pytest.mark.parametrize("m, n, k", [(128, 64, 64), (256, 128, 128), (128, 128, 128)])
def test_fp4_selector_reports_at_least_one_xcd(m, n, k):
    assert _selector(m, n, k, "f4").num_sms >= 1
