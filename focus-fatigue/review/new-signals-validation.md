# Validation Report: New Signals

**Validator:** Automated Validation Agent  
**Date:** 2026-07-21  
**Test file:** `fixtures/test_new_signals.py`  

---

## Test Execution

**Command:** `python fixtures/test_new_signals.py`  
**Result:** ✅ ALL 20 TESTS PASSED  
**Execution time:** 3.167 seconds  

---

## Test Results Detail

| Test | Status | Notes |
|------|--------|-------|
| `test_polarisation_output_schema` | ✅ PASS | All 8 OUTPUT_COLUMNS present, correct signal_name |
| `test_polarisation_values_in_0_1` | ✅ PASS | signal_value ∈ [0, 1] |
| `test_polarisation_aligned_team_near_1` | ✅ PASS | Mean R > 0.95 for aligned motion |
| `test_polarisation_random_team_near_0` | ✅ PASS | Mean R < 0.5 for random motion |
| `test_polarisation_no_blocks_returns_empty` | ✅ PASS | Empty DataFrame returned |
| `test_polarisation_missing_velocity_columns` | ✅ PASS | Zero-velocity fallback warning logged |
| `test_polarisation_handles_missing_possession` | ✅ PASS | NaN possession handled |
| `test_polarisation_team_level_output` | ✅ PASS | player_id = 0 (team-level) |
| `test_polarisation_validate_passes` | ✅ PASS | validate() returns True |
| `test_polarisation_validate_empty` | ✅ PASS | validate() handles empty DataFrame |
| `test_centroid_distance_output_schema` | ✅ PASS | All 8 OUTPUT_COLUMNS present |
| `test_centroid_distance_non_negative` | ✅ PASS | All distances ≥ 0 |
| `test_centroid_distance_no_ball_player` | ✅ PASS | player_id = -1 filtered out |
| `test_centroid_distance_plausible_magnitude` | ✅ PASS | Max distance ≤ 150m |
| `test_centroid_distance_empty_blocks` | ✅ PASS | Empty DataFrame returned |
| `test_centroid_distance_missing_x_y` | ✅ PASS | KeyError raised (expected) |
| `test_centroid_distance_handles_missing_possession` | ✅ PASS | Both teams processed as OOP |
| `test_centroid_distance_per_player_output` | ✅ PASS | Multiple unique player_ids |
| `test_centroid_distance_validate_passes` | ✅ PASS | validate() returns True |
| `test_centroid_distance_validate_empty` | ✅ PASS | validate() handles empty DataFrame |

---

## Edge Cases Tested

| Edge Case | Result |
|-----------|--------|
| Empty blocks | ✅ Graceful empty DataFrame |
| Missing velocity columns (polarisation) | ✅ Warning + zero-velocity fallback |
| Missing x/y columns (centroid) | ✅ KeyError (expected, cannot compute) |
| All-NaN team_in_possession | ✅ No crash, treats both teams as OOP |
| Empty DataFrame validation | ✅ Returns True |

---

## Code Quality Checks

| Check | Result |
|-------|--------|
| No imports from `src.model1` | ✅ PASS |
| No imports from `src.config` | ✅ PASS |
| Extends `SignalBase` from `src.signals.base` | ✅ PASS |
| Uses `@register_signal` from `src.signals.registry` | ✅ PASS |
| Tests run without real data (synthetic only) | ✅ PASS |
| Tests pass on Pi environment | ✅ PASS |

---

## Review Findings Addressed

| Issue | Status |
|-------|--------|
| M1 — `_blocks_to_dicts` duplication | ⚠️ Open (not a correctness issue) |
| L1 — Redundant frame-mask conditions | ✅ Fixed before validation |
| L2 — Unused `DEFAULT_SIGNAL_CONFIG` import | ✅ Fixed before validation |
| L3 — Missing SIGNAL_DESCRIPTIONS | ✅ Fixed before validation |

---

## Verdict

**PASS** ✅ — All 20 tests pass. The code is ready to commit.

No regressions introduced. Both signals produce correct output schema, correct value ranges, and handle edge cases gracefully. The open MEDIUM issue (code duplication) does not affect correctness and can be addressed in a follow-up.
