# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
"""
Shared GPU/CU utilities for TritonBLAS benchmarks.
"""
import ctypes

import torch  # type: ignore


def get_num_xcds(device_id: int = 0) -> int:
    """Query the number of XCDs (chiplets) via HIP runtime."""
    try:
        hip = ctypes.cdll.LoadLibrary("libamdhip64.so")
    except OSError:
        return 1
    try:
        hipDeviceAttributeNumberOfXccs = 10018
        xcc_count = ctypes.c_int()
        hip.hipDeviceGetAttribute(ctypes.byref(xcc_count), hipDeviceAttributeNumberOfXccs, device_id)
        return xcc_count.value
    except Exception:
        return 1


def get_cu_info(device_id: int = 0):
    """Return (total_cus, num_xcds, cus_per_xcd) for the current device."""
    total_cus = torch.cuda.get_device_properties(device_id).multi_processor_count
    num_xcds = get_num_xcds(device_id)
    cus_per_xcd = total_cus // num_xcds
    return total_cus, num_xcds, cus_per_xcd


def build_balanced_hex_mask(remove_per_xcd: int, num_xcds: int, cus_per_xcd: int) -> str:
    """Build a ROC_GLOBAL_CU_MASK hex string that removes CUs from the top of every XCD."""
    if remove_per_xcd == 0:
        return ""
    total_cus = num_xcds * cus_per_xcd
    mask = (1 << total_cus) - 1
    for i in range(remove_per_xcd):
        base = num_xcds * i
        for xcd in range(num_xcds):
            bit = base + xcd
            mask &= ~(1 << bit)
    return f"0x{mask:x}"
