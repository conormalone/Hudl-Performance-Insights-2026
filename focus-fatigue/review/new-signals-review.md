# Code Review: New Signals — Team Polarisation & Team Centroid Distance

**Reviewer:** Automated Review Agent  
**Date:** 2026-07-21  
**Scope:** All files modified/created for the two new signals  

---

## Files Reviewed

| File | Status |
|------|--------|
| `src/signals/polarisation.py` | NEW |
| `src/signals/team_centroid_distance.py` | NEW |
| `src/pipeline.py` | MODIFIED (2 imports + 2 descriptions) |
| `notebooks/run_new_signals.ipynb` | NEW |
| `fixtures/test_new_signals.py` | NEW |
| `analysis/opponent_quality_covariate.py` | NEW |

---

## Findings Summary

| Severity | Count |
|----------|-------|
| **HIGH** | **0** |
| **MEDIUM** | **1** |
| **LOW** | **3** |

---

## HIGH — None

No HIGH severity issues found.

---

## MEDIUM

### M1 — `_blocks_to_dicts()` is duplicated in both signal files

- **File:** `src/signals/polarisation.py` (lines 218–234), `src/signals/team_centroid_distance.py` (lines 184–200)
- **Severity:** MEDIUM
- **Description:** The identical helper function `_blocks_to_dicts` is defined in both signal modules. If the block conversion logic ever changes (e.g. a new block format is introduced), both copies must be updated, creating a maintenance hazard.
- **Recommendation:** Extract to a shared utility, e.g. `src.signals.utils` or inline into a method on `SignalBase`. For now, acceptable with a comment noting the duplication.

---

## LOW

### L1 — Redundant frame-range mask conditions

- **Files:** `src/signals/polarisation.py` lines 147–150, `src/signals/team_centroid_distance.py` lines 116–119
- **Severity:** LOW
- **Description:** The frame mask applies both `pd.Series.between(start, end, inclusive="left")` AND separate `>= start` / `< end` conditions. `between(…, inclusive="left")` already implements `[start, end)`, making the standalone comparisons redundant.
- **Recommendation:** Remove the redundant conditions. Not a correctness issue — the mask is correct either way.

**Current code:**
```python
frame_mask = (
    match_df["frame_count"].between(start, end, inclusive="left")
    & (match_df["frame_count"] >= start)
    & (match_df["frame_count"] < end)
)
```

**Simplified:**
```python
frame_mask = match_df["frame_count"].between(start, end, inclusive="left")
```

### L2 — Unused import `DEFAULT_SIGNAL_CONFIG` in polarisation.py

- **File:** `src/signals/polarisation.py` line 6
- **Severity:** LOW
- **Description:** `from .config import DEFAULT_SIGNAL_CONFIG` is imported but never used directly — it's only passed through to `super().__init__()` which already defaults to `DEFAULT_SIGNAL_CONFIG` when `config=None`.
- **Recommendation:** Remove the unused import.

### L3 — `SIGNAL_DESCRIPTIONS` dict updated after code review (was missing initially)

- **File:** `src/pipeline.py`
- **Severity:** LOW
- **Description:** The `SIGNAL_DESCRIPTIONS` dictionary did not initially include entries for `team_polarisation` or `team_centroid_distance`, causing `list-signal-descriptions` to show blank descriptions. This was caught and fixed during the review cycle.
- **Recommendation:** Already fixed. No further action needed.

---

## Detailed Per-File Assessment

### `src/signals/polarisation.py`

- ✅ Extends `SignalBase` from `src.signals.base`
- ✅ Uses `@register_signal` decorator from `src.signals.registry`
- ✅ `signal_name = "team_polarisation"`
- ✅ `compute()` method accepts `(match_df, blocks, *, game_id=...)` correctly
- ✅ Handles both `list[DataFrame]` and `list[dict]` block formats
- ✅ `validate()` calls `super().validate()` then checks [0, 1] range
- ✅ `ensure_output_columns()` called before returning
- ✅ Type casting correct (int columns, str columns, float signal_value)
- ✅ Per-frame vectorised polarisation computation (no Python loops per player)
- ✅ Edge cases: empty match, empty blocks, missing velocity columns, all-NaN possession
- ✅ Issue L1 (redundant mask) — LOW
- ✅ Issue L2 (unused import) — LOW

### `src/signals/team_centroid_distance.py`

- ✅ Extends `SignalBase` from `src.signals.base`
- ✅ Uses `@register_signal` decorator from `src.signals.registry`
- ✅ `signal_name = "team_centroid_distance"`
- ✅ `compute()` method accepts `(match_df, blocks, *, game_id=...)` correctly
- ✅ Handles both block formats
- ✅ `validate()` calls `super().validate()` then checks non-negative & pitch bounds
- ✅ `ensure_output_columns()` called before returning
- ✅ Per-player output (standard signal pattern)
- ✅ Filters ball player (id=-1) correctly
- ✅ Edge cases covered
- ✅ Issue L1 (redundant mask) — LOW
- ✅ Issue M1 (code duplication shared with polarisation) — MEDIUM

### `src/pipeline.py`

- ✅ Two import lines added after `physical_load` (line 35)
- ✅ Import comments match signal names
- ✅ Descriptions added to `SIGNAL_DESCRIPTIONS` (L3, already fixed)
- ✅ No conflicts with existing imports or logic

### `notebooks/run_new_signals.ipynb`

- ✅ `sys.path` set to project root
- ✅ Imports match existing project structure (no `src.model1`, no `src.config`)
- ✅ `list_signals()` assertion verifies registration
- ✅ Helper function for single-match processing
- ✅ Merge step calls `merge_all()`
- ✅ Cells clearly documented
- ✅ Works with both sample and full tracking directories

### `fixtures/test_new_signals.py`

- ✅ Synthetic data (no real data files needed)
- ✅ Tests output schema (all 8 OUTPUT_COLUMNS)
- ✅ Polarisation: aligned team R ≈ 1, random team R near 0
- ✅ Centroid distance: non-negative, no ball player, pitch bounds
- ✅ Empty blocks → empty DataFrame
- ✅ Missing columns → graceful handling (warning/fallback)
- ✅ `validate()` on empty and non-empty outputs
- ✅ Uses unittest (runs on Pi)

### `analysis/opponent_quality_covariate.py`

- ✅ Loads unified dataset from Parquet
- ✅ Builds opponent quality from fixtures CSV (goals scored per match)
- ✅ Uses shape JSON metadata for team name resolution
- ✅ Falls back to heuristic when shape files unavailable
- ✅ Script-style (argparse) with sensible defaults
- ✅ No dependency on `src.model1` or `src.config`

---

## Verdict

**Recommendation: ACCEPT after addressing MEDIUM issue M1 (code duplication).**

The duplicated `_blocks_to_dicts` function is not a correctness issue but should be consolidated in a follow-up. No HIGH issues were found. All critical rules (no `src.model1`, no `src.config`, proper SignalBase extension, proper registration) are satisfied.

The implementation is production-ready for commit and testing on real data.
