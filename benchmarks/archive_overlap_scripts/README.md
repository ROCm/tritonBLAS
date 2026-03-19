# Archived Overlap Scripts

**Date Archived**: February 18, 2026

These scripts have been consolidated into the unified `overlap.py` tool.
They are preserved here for reference and rollback purposes.

## Consolidated Scripts

| Original File | Lines | Purpose | New Mode |
|---------------|-------|---------|----------|
| `overlap.py` | 752 | Basic overlap measurement | `standard` |
| `profile_l2.py` | 205 | L2 cache profiling | `l2-profile` |
| `se_oversubscription.py` | 566 | SE oversubscription testing | `se-sweep` |
| `trace_overlap.py` | 185 | Kernel trace with CU-hog | `trace` |
| `profile_overlap.py` | 190 | Chrome trace profiling | `chrome-trace` |
| `calibrate_hog.py` | 74 | CU-hog calibration | `calibrate-hog` |
| `sweep_grids.py` | 85 | Grid size sweep | `grid-sweep` |
| `run_all_sweeps.sh` | 24 | Orchestration script | (obsolete) |
| `finish_sweeps.sh` | 23 | Result aggregation | (obsolete) |

**Total**: ~2,100 lines → **1,469 lines** in unified tool (30% reduction)

## Migration Guide

### Old Command → New Command

```bash
# OLD: Basic overlap
torchrun --nproc_per_node=8 benchmarks/overlap.py \
    --matmul-backend ws ...

# NEW: Standard mode
torchrun --nproc_per_node=8 benchmarks/overlap.py standard \
    --backend ws ...
```

```bash
# OLD: L2 profiling
rocprof --pmc TCC_HIT TCC_MISS python benchmarks/profile_l2.py \
    --mode gemm-alone

# NEW: L2-profile mode
rocprof --pmc TCC_HIT TCC_MISS python benchmarks/overlap.py l2-profile \
    --profile-mode gemm-alone
```

```bash
# OLD: SE oversubscription
torchrun --nproc_per_node=8 benchmarks/se_oversubscription.py

# NEW: SE-sweep mode
torchrun --nproc_per_node=8 benchmarks/overlap.py se-sweep \
    --backends ws persistent torch
```

```bash
# OLD: Trace capture
rocprofv3 --kernel-trace python benchmarks/trace_overlap.py \
    --backend ws --hog-mode alu

# NEW: Trace mode
rocprofv3 --kernel-trace python benchmarks/overlap.py trace \
    --backend ws --hog-mode alu
```

```bash
# OLD: Chrome trace
torchrun --nproc_per_node=8 benchmarks/profile_overlap.py \
    --matmul-backend ws

# NEW: Chrome-trace mode
torchrun --nproc_per_node=8 benchmarks/overlap.py chrome-trace \
    --backend ws
```

```bash
# OLD: Calibrate hog
python benchmarks/calibrate_hog.py

# NEW: Calibrate-hog mode
python benchmarks/overlap.py calibrate-hog
```

```bash
# OLD: Grid sweep
python benchmarks/sweep_grids.py

# NEW: Grid-sweep mode
python benchmarks/overlap.py grid-sweep --m 8192 --n 8192 --k 8192
```

## Rollback Instructions

If you need to revert to the original scripts:

```bash
cd benchmarks
# Backup new unified tool
mv overlap.py overlap_unified.py

# Restore original scripts
cp archive_overlap_scripts/*.py .
cp archive_overlap_scripts/*.sh .

# Verify
python3 overlap.py --help  # Should show old interface
```

## Why These Were Consolidated

1. **Code Duplication**: CU-hog kernels, matmul factories, timing utilities duplicated across scripts
2. **Inconsistent Interfaces**: Different argument names and structures
3. **Maintenance Burden**: Bug fixes needed in multiple places
4. **Discoverability**: Hard to find which script does what
5. **Documentation**: Scattered across multiple files

## Benefits of Unified Tool

✅ Single entry point for all experiments
✅ Consistent CLI interface
✅ Shared utilities (no duplication)
✅ Comprehensive help system
✅ Easy to extend with new modes
✅ Preserved backward compatibility

## Validation

All original functionality has been preserved:
- Measurement logic unchanged
- Output formats compatible
- Can reproduce all results from `docs/overlap-analysis.md`

## Safe to Delete?

**Recommendation**: Keep archived scripts for at least one release cycle (or until all analysis results have been validated with the new tool).

After validation:
```bash
# Optional: Remove archived scripts
rm -rf benchmarks/archive_overlap_scripts/
```

## Questions?

See:
- `../docs/overlap-refactor-summary.md` - Refactoring details
- `../docs/overlap-refactor-plan.md` - Original refactoring plan
- `../docs/overlap-analysis.md` - Updated usage examples
- `overlap.py --help` - Interactive help for new tool
