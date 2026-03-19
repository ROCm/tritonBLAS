# Overlap Benchmark Refactoring Plan

## Goal
Consolidate all overlap analysis functionality from scattered scripts into a single, clean, comprehensive `overlap.py` benchmark tool with all experiments exposed through CLI arguments.

## Current State Analysis

### Untracked Files to Consolidate
1. **overlap.py** (752 lines) - Main overlap benchmark (keep as base)
2. **profile_l2.py** (205 lines) - L2 cache profiling with rocprof
3. **se_oversubscription.py** - SE oversubscription test suite  
4. **trace_overlap.py** (185 lines) - Kernel trace capture with CU-hog
5. **profile_overlap.py** - Chrome trace profiling
6. **calibrate_hog.py** (74 lines) - CU-hog kernel calibration
7. **sweep_grids.py** (85 lines) - Grid size sweep (isolated)
8. **parse_l2_counters.py** (57 lines) - Parse rocprof output
9. **parse_trace.py** (93 lines) - Parse rocprofv3 traces
10. **parse_rccl_trace.py** (140 lines) - Parse RCCL traces
11. **run_all_sweeps.sh** (24 lines) - Orchestration script
12. **finish_sweeps.sh** (23 lines) - Result aggregation
13. **plot_cu_sweep.py** - CU sweep plotting (just created)

### Core Features to Support

#### 1. **Baseline Overlap Measurement** (existing in overlap.py)
- GEMM alone, Comm alone, Overlapped
- Support for rotating buffers (cold L2) vs warm cache
- Multiple backends: torch, persistent, streamk, ws, ws-global
- Multiple collectives: all_reduce, all_gather, all_to_all
- Per-iteration timing with statistics (min/max/mean/median)
- CSV output with comprehensive metrics

#### 2. **L2 Cache Analysis** (from profile_l2.py)
- Run with rocprof hardware counters (TCC_HIT, TCC_MISS, TCC_READ, TCC_WRITEBACK)
- Modes:
  - `gemm-alone`: Isolated GEMM
  - `gemm-polluted`: GEMM + memory streaming
  - `gemm-rccl`: GEMM + actual RCCL (distributed)
- Support for pollution buffer size configuration

#### 3. **SE Oversubscription Testing** (from se_oversubscription.py)
- Shape sweep across trigger/safe configurations
- Measure both warm and rotating baselines
- Report mean + max (excluding first iteration)
- Include Tensile tile information for torch backend
- Detect SE alignment issues (% 32, % 8)

#### 4. **Kernel Trace Capture** (from trace_overlap.py)
- rocprofv3 kernel-trace mode
- CU-hog modes: ALU (pure compute) vs MEM (memory streaming)
- Configurable hog parameters (WGs, iterations)
- Compare alone vs overlapped

#### 5. **Chrome Trace Profiling** (from profile_overlap.py)
- torch.profiler integration
- Export Chrome trace JSON for visualization
- Key averages table
- Rank 0 only (distributed aware)

#### 6. **CU-Hog Calibration** (from calibrate_hog.py)
- Standalone mode to calibrate hog kernel iterations
- Test both ALU and MEM modes
- Output target duration mappings

#### 7. **Grid Size Sweep** (from sweep_grids.py)
- Isolated single-GPU test
- Sweep work-stealing over different CU counts
- Measure performance vs grid size

## Proposed Architecture

### Command Structure
```
overlap.py <mode> [options]
```

### Modes

#### Mode 1: `standard` (default)
The existing overlap benchmark - GEMM alone, Comm alone, Overlapped
```bash
torchrun --nproc_per_node=8 overlap.py standard \
    --backend ws \
    --m 8192 --n 8192 --k 8192 \
    --comm-size 8192 8192 \
    --collective all_reduce \
    --rotating-buffers \
    --steps 200 \
    --output-csv results.csv
```

**Arguments:**
- `--backend`: torch | persistent | streamk | ws | ws-global
- `--m, --n, --k`: GEMM dimensions
- `--comm-size`: Communication tensor shape
- `--collective`: all_reduce | all_gather | all_to_all
- `--rotating-buffers`: Enable rotating buffer baseline (recommended)
- `--warm-cache`: Also measure warm cache baseline
- `--nccl-max-nchannels`: NCCL channel configuration
- `--steps`: Number of timed iterations
- `--warmup`: Warmup iterations
- `--output-csv`: CSV output path
- `--serial-test`: Also run serial test (NCCL finishes, then GEMM)
- `--cu-hog-test`: Also run with CU-hog instead of RCCL (see below)

#### Mode 2: `l2-profile`
Run for L2 cache hardware counter collection (wrap with rocprof)
```bash
rocprof --pmc TCC_HIT TCC_MISS TCC_READ TCC_WRITEBACK \
    python overlap.py l2-profile \
        --profile-mode gemm-alone \
        --backend ws \
        --m 8192 --n 8192 --k 8192

rocprof --pmc TCC_HIT TCC_MISS TCC_READ TCC_WRITEBACK \
    torchrun --nproc_per_node=8 overlap.py l2-profile \
        --profile-mode gemm-rccl \
        --backend ws \
        --m 8192 --n 8192 --k 8192 \
        --comm-size 8192 8192
```

**Arguments:**
- `--profile-mode`: gemm-alone | gemm-polluted | gemm-rccl
- `--pollution-mb`: Size of pollution buffer (for gemm-polluted mode)
- All standard backend/size arguments

#### Mode 3: `se-sweep`
SE oversubscription shape sweep
```bash
torchrun --nproc_per_node=8 overlap.py se-sweep \
    --backends ws persistent torch \
    --shapes-preset small   # or 'large' or 'all' or custom list
    --rotating-buffers \
    --steps 100 \
    --output-csv se_results.csv
```

**Arguments:**
- `--backends`: List of backends to test
- `--shapes-preset`: small | large | all | custom
- `--custom-shapes`: JSON list of (M,N,K) tuples
- `--rotating-buffers`: Use rotating baseline
- `--warm-cache`: Also measure warm cache
- `--include-tensile-info`: Add Tensile tile information to output

#### Mode 4: `trace`
Kernel trace capture with CU-hog (wrap with rocprofv3)
```bash
rocprofv3 --kernel-trace -d /tmp/trace_ws_alu -f csv \
    python overlap.py trace \
        --backend ws \
        --m 8192 --n 8192 --k 8192 \
        --hog-mode alu \
        --steps 5

# Standalone GEMM (no overlap)
rocprofv3 --kernel-trace -d /tmp/trace_alone -f csv \
    python overlap.py trace \
        --backend ws \
        --m 8192 --n 8192 --k 8192 \
        --no-overlap \
        --steps 5
```

**Arguments:**
- `--hog-mode`: alu | mem
- `--no-overlap`: Run GEMM alone (no CU-hog)
- `--hog-wgs`: Number of workgroups for hog
- `--hog-alu-iters`: ALU iterations
- `--hog-mem-iters`: Memory iterations
- All standard backend/size arguments

#### Mode 5: `chrome-trace`
Chrome trace profiling with torch.profiler
```bash
torchrun --nproc_per_node=8 overlap.py chrome-trace \
    --backend ws \
    --m 8192 --n 8192 --k 8192 \
    --comm-size 4096 4096 \
    --steps 5 \
    --output-dir /tmp/overlap_profile
```

**Arguments:**
- `--output-dir`: Directory for trace files
- All standard backend/size/collective arguments

#### Mode 6: `calibrate-hog`
Calibrate CU-hog kernels (single GPU, non-distributed)
```bash
python overlap.py calibrate-hog
```

**Output:** Prints iteration counts needed for various durations

#### Mode 7: `grid-sweep`
Grid size sweep for work-stealing (single GPU)
```bash
python overlap.py grid-sweep \
    --m 8192 --n 8192 --k 8192 \
    --grid-sizes 304 296 288 280 272 256 240 224 200 176 152 128 \
    --steps 50
```

**Arguments:**
- `--grid-sizes`: List of grid sizes (WG counts) to test
- Standard GEMM size arguments

## File Organization

### Main File: `overlap.py` (refactored)
```
overlap.py
├── Imports & Constants
├── Utility Functions
│   ├── CU-hog kernels (ALU & MEM)
│   ├── Matmul backend factories
│   └── Statistics helpers
├── Core Measurement Functions
│   ├── _time_per_iter (warm)
│   ├── _time_rotating (cold)
│   ├── _time_serial (post-RCCL)
│   └── _time_overlapped
├── Mode Implementations
│   ├── mode_standard()
│   ├── mode_l2_profile()
│   ├── mode_se_sweep()
│   ├── mode_trace()
│   ├── mode_chrome_trace()
│   ├── mode_calibrate_hog()
│   └── mode_grid_sweep()
├── Argument Parsing
│   ├── add_common_args()
│   ├── add_gemm_args()
│   ├── add_comm_args()
│   └── mode-specific parsers
└── main()
```

### Support Files to Keep
- `plot_cu_sweep.py` - Standalone plotting utility (separate responsibility)
- Potentially: `parse_*.py` scripts IF they're used as utilities (can consolidate later)

### Files to Remove After Refactor
- `profile_l2.py`
- `se_oversubscription.py`
- `trace_overlap.py`
- `profile_overlap.py`
- `calibrate_hog.py`
- `sweep_grids.py`
- `run_all_sweeps.sh`
- `finish_sweeps.sh`

## Implementation Steps

### Phase 1: Code Organization
1. Create new file structure with mode dispatch
2. Extract shared utilities (CU-hog, matmul factories, timing helpers)
3. Preserve existing `standard` mode functionality (current overlap.py)

### Phase 2: Mode Integration
4. Integrate `l2-profile` mode (from profile_l2.py)
5. Integrate `se-sweep` mode (from se_oversubscription.py)
6. Integrate `trace` mode (from trace_overlap.py)
7. Integrate `chrome-trace` mode (from profile_overlap.py)
8. Integrate `calibrate-hog` mode (from calibrate_hog.py)
9. Integrate `grid-sweep` mode (from sweep_grids.py)

### Phase 3: Argument Parsing
10. Create hierarchical argument parser with subcommands
11. Share common arguments across modes
12. Add mode-specific argument groups
13. Validate argument combinations

### Phase 4: Testing & Documentation
14. Test each mode against original scripts
15. Update overlap-analysis.md with new usage examples
16. Create comprehensive --help documentation
17. Add examples section to docstring

### Phase 5: Cleanup
18. Remove old scripts
19. Update any references in docs
20. Clean up git status

## Key Design Decisions

### 1. Distributed vs Single-GPU Detection
- Auto-detect distributed environment (check `RANK` env var or `dist.is_initialized()`)
- Some modes require distributed (standard with real RCCL, se-sweep, chrome-trace, l2-profile with gemm-rccl)
- Some modes are single-GPU only (grid-sweep, calibrate-hog, l2-profile gemm-alone/gemm-polluted, trace)
- Validate at mode entry point

### 2. Rotating Buffer Implementation
- Keep existing rotating buffer logic from overlap.py
- Make it a flag for all modes that support it
- Default to rotating (recommended) with option to also measure warm

### 3. Output Formats
- CSV for tabular results (standard, se-sweep)
- JSON for structured data (potential future use)
- Chrome trace JSON (chrome-trace mode)
- rocprof text output (captured externally for l2-profile, trace)
- Console tables for interactive use

### 4. Backward Compatibility
- Keep existing CSV column names for standard mode
- Ensure old analysis scripts can still parse output
- Add version field to CSV header

### 5. Extensibility
- Use mode dispatch dictionary for easy addition of new modes
- Shared argument groups for common parameters
- Factory pattern for backend creation
- Clear separation of measurement logic from I/O

## Success Criteria

1. ✅ All functionality from 13 untracked files is accessible through single script
2. ✅ Each experiment type has a clear mode with intuitive arguments
3. ✅ No code duplication between modes
4. ✅ Output format compatible with existing analysis
5. ✅ Documentation updated with new usage patterns
6. ✅ Clean git status after removing consolidated files
7. ✅ Can reproduce all results from overlap-analysis.md
8. ✅ New experiments are easy to add without modifying existing modes

## Example Workflows

### Workflow 1: Basic Overlap Measurement
```bash
# Rotating baseline (recommended)
torchrun --nproc_per_node=8 overlap.py standard \
    --backend ws --m 8192 --n 8192 --k 8192 \
    --comm-size 8192 8192 --collective all_reduce \
    --rotating-buffers --steps 200 \
    --output-csv ws_overlap.csv

# Compare backends
for backend in ws persistent torch; do
    torchrun --nproc_per_node=8 overlap.py standard \
        --backend $backend \
        --m 8192 --n 8192 --k 8192 \
        --comm-size 8192 8192 --collective all_reduce \
        --rotating-buffers --steps 200 \
        --output-csv ${backend}_overlap.csv
done
```

### Workflow 2: L2 Cache Investigation
```bash
# 1. Measure GEMM alone with counters
rocprof --pmc TCC_HIT TCC_MISS python overlap.py l2-profile \
    --profile-mode gemm-alone --backend ws \
    --m 8192 --n 8192 --k 8192 --steps 50

# 2. Measure with RCCL overlap
rocprof --pmc TCC_HIT TCC_MISS torchrun --nproc_per_node=8 overlap.py l2-profile \
    --profile-mode gemm-rccl --backend ws \
    --m 8192 --n 8192 --k 8192 --comm-size 8192 8192 --steps 50

# 3. Parse results (separate utility)
python parse_l2_counters.py rocprof_output.csv
```

### Workflow 3: SE Oversubscription Study
```bash
torchrun --nproc_per_node=8 overlap.py se-sweep \
    --backends ws persistent torch \
    --shapes-preset all \
    --rotating-buffers --warm-cache \
    --steps 100 \
    --output-csv se_sweep_results.csv
```

### Workflow 4: Kernel Timeline Analysis
```bash
# Capture trace with CU-hog
rocprofv3 --kernel-trace -d /tmp/trace_ws_alu -f csv \
    python overlap.py trace \
        --backend ws --m 8192 --n 8192 --k 8192 \
        --hog-mode alu --steps 5

# Parse trace
python parse_trace.py /tmp/trace_ws_alu/kernel_trace.csv
```

## Notes

- Keep all timing logic consistent with current overlap.py
- Preserve CUDA event timing for accuracy
- Maintain compatibility with MI300X hardware detection
- Ensure NCCL environment variables are respected
- Add helpful error messages for invalid mode/argument combinations
- Consider adding a `--dry-run` flag to validate arguments without running
