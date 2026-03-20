# DF Perfmon Counters on MI300X: Privileged CP Queues Required

## Summary

Data Fabric performance counters (`MALL_BANDWIDTH_ALL`, `HBM_READ_BYTES`, `HBM_WRITE_BYTES`) collected via `rocprofv3` require **privileged command processor queues** to be enabled on MI300X. Without this setting, programming DF PerfMon registers via PConfig packets causes an immediate GPU hang and XGMI hive-wide reset.

## Fix

```bash
# As root — enable privileged CP queues
echo 1 > /sys/module/amdgpu/parameters/priv_cp_queues
```

This must be set **before** running `rocprofv3` with DF counters. It persists until reboot. To make it permanent, add `amdgpu.priv_cp_queues=1` to the kernel command line or a modprobe config.

## Verified Working

After enabling `priv_cp_queues`, all DF and L2 counters work correctly:

| Counter | Status | Description |
|---------|--------|-------------|
| `MALL_BANDWIDTH_ALL` | Working | MALL/LLC total bandwidth |
| `HBM_READ_BYTES` | Working | HBM read traffic |
| `HBM_WRITE_BYTES` | Working | HBM write traffic |
| `TCC_HIT_sum` | Working | L2 cache hits (works without priv queues too) |
| `TCC_MISS_sum` | Working | L2 cache misses (works without priv queues too) |
| `TCC_EA0_RDREQ_DRAM_sum` | Working | L2→DRAM read requests (works without priv queues too) |
| `TCC_EA0_WRREQ_DRAM_sum` | Working | L2→DRAM write requests (works without priv queues too) |

## Why This Happens

The DF perfmon path in `aqlprofile` uses `BuildWritePConfigRegPacket` to write to DF PerfMon control/counter registers via SMN addresses (base `0x49000000`, offsets `0x0C80`+). These PConfig writes target the Data Fabric IP block, which sits outside the normal GPU shader/memory controller address space. Without privileged CP queues, the command processor rejects these writes, causing a GPU hang.

Standard TCC/SQ counters (L2 cache, shader) don't need privileged queues because they access registers within the GPU's normal perfmon address space.

## Reproducer (Without Fix — Will Hang)

```bash
# This WILL hang without priv_cp_queues=1
rocprofv3 \
    --pmc MALL_BANDWIDTH_ALL \
    -o out -d df --output-format csv \
    -- hip-stream -s 102400 -n 3
```

## System Information

| Component | Value |
|-----------|-------|
| **GPU** | AMD Instinct MI300X HF (8x, XGMI fully connected) |
| **Device ID** | 0x74a9 |
| **GFX Target** | gfx942 (gfx_target_version 90402) |
| **Compute Partition** | NPS1, SPX |
| **OS** | Ubuntu 22.04.4 LTS |
| **Kernel** | 5.15.0-171-generic |
| **amdgpu driver** | 6.16.13 (also reproduced on 6.9.4) |
| **Platform** | Microsoft C278A blade, 2x Intel Xeon Platinum 8480C |

## aqlprofile Build

- **Source**: `AMD-ROCm-Internal/rocm-systems` PR #452
- **Branch**: `users/mkuriche/umc-df-ipdiscovery`
- **HEAD commit**: `765f38daa4`
- **Key commits**:
  - `93f1f18193` — [aqlprofile] Add DF perfmon support for MI300
  - `9f39d2d6b6` — [aqlprofile] Update DF counters encoding scheme
  - `42284de515` — [aqlprofile] DF perfmon: Reset both control registers

## Notes

- With driver 6.16.13, the XGMI reset **succeeds** and GPUs recover after a hang. With the older driver 6.9.4, the reset entered an infinite loop in `psp_wait_for_bootloader_steady_state` and required a hard reboot.
- `HSA_ENABLE_SDMA=0` is also needed in the container when using the custom `rocr-runtime` with driver 6.16.13.
- The `amdgpu-dkms-firmware` package (6.9.4) is older than the driver (6.16.13).
