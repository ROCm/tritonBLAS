# Overlap Benchmark Testing Report

## Test Summary: All Single-GPU Modes PASS ✅

### Modes Tested Successfully:
1. ✅ **calibrate-hog** - Working perfectly
2. ✅ **grid-sweep** - Fixed and working  
3. ✅ **trace (no-overlap)** - Working perfectly
4. ✅ **trace (with CU-hog)** - Working perfectly
5. ✅ **l2-profile (gemm-alone)** - Working perfectly

### Issue Fixed:
- **Grid-sweep mode**: Changed from passing total_cus to monkey-patching selector._hardware.N_CU

### Multi-GPU Modes (Require torchrun):
- standard, se-sweep, chrome-trace, l2-profile (gemm-rccl)
- These need 8-GPU setup with NCCL for full testing

### Status: Ready for use ✅
All testable single-GPU modes validated and working correctly.
