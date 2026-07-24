# Integration Audit Report

**Date:** 13 July 2026
**Status:** ALL ISSUES FIXED ✅

## Issues Found & Resolution

### BLOCKER Issues

| ID | Issue | Status | Fix |
|----|-------|--------|-----|
| B1 | No bulk runner for all signals | ✅ FIXED | Created `src/run_signals.py` — CLI script iterates over matches, runs all registered signals via SIGNAL_REGISTRY. Supports `--all`, `--match`, `--signal` args. |
| B2 | No unified pipeline (Model 1 → Model 2) | ✅ FIXED | Created `src/run_pipeline.py` — runs Model 1 first (frame processing → blocks → window aggregates), then runs all signals. Single command: `python3 src/run_pipeline.py --all`. |
| B3 | Hardcoded paths in 7 files | ✅ FIXED | `src/config.py` uses `Path(__file__).resolve()` for all paths. No `/home/conormalone/...` paths remain. `config.yaml` provides YAML-based overrides. `COGLOAD_DATA` and `COGLOAD_OUTPUT` env vars for runtime override. |
| B4 | No sample data in repo | ✅ FIXED | Created `src/download_data.py` — generates synthetic fixtures with `--fixture` flag. `fixtures/sample_block.py` provides known-output fixture for heading validation. README documents data requirements. |
| B5 | `heading` column missing from smoothing | ✅ FIXED | `smoothing.py::compute_velocity_features()` now produces `heading` (radians in [0, 2π)) and `heading_deg` (degrees in [0, 360)) via `np.arctan2(vy, vx)`. Validated with test fixture yielding exact expected angles. |
| B6 | Block format inconsistency | ✅ FIXED | All `run_pipeline()` methods accept `list[pd.DataFrame]`. `aggregate_shift_latency_by_block` accepts and returns `pd.DataFrame`. `validate_signals.check_block_format_consistency()` validates all blocks are DataFrames. |

### HIGH Issues

| ID | Issue | Status | Fix |
|----|-------|--------|-----|
| H7 | No output merge script | ✅ FIXED | Created `src/merge_outputs.py` — reads all signal CSVs + Model 1 CSVs, handles wide-vs-long format difference, merges into unified dataset. |
| H8 | No README | ✅ FIXED | `README.md` rewritten with: project description, data requirements, installation, usage, output structure, dependencies, and project structure. |

## Validation Results

- 26 Python source files, 3,588 lines of code
- All files compile without syntax errors
- Heading computation validated: 8/8 test cases pass (0°, 90°, 180°, 270°, 45°, 135°, 225°, 315°)
- No hardcoded paths in any source file
- Full module tree findable via Python import system
- Git repo initialised and committed (main branch)

## How to Use

```bash
# Install dependencies
pip install -r requirements.txt

# Generate test data
python3 src/download_data.py --fixture

# Run the full pipeline
python3 src/run_pipeline.py --all --nrows 6000

# Validate everything
python3 src/validate_all.py
```
