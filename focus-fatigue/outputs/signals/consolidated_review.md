# Consolidated Code Review — Signal Pipeline v2

**Reviewer:** Project Manager 📋  
**Date:** 2026-07-13  
**Scope:** Signals 1, 3, 5 + Integration fixes (Model 1 H-fixes)

---

## Per-Signal Findings

### Signal 5 — Transition Recognition (`src/signals/transition/`)

| ID | Severity | File | Finding | Recommendation |
|----|----------|------|---------|----------------|
| S5-H1 | **HIGH** | `detector.py:120` | `classify_transition_type` multiplies ball speed by `frames_per_second`. Velocities are already in **m/s** (confirmed via `smoothing.py` — `np.gradient / dt`). A ball at 20 m/s is reported as 500 m/s → nearly all transitions classified as "surprise". | Remove `* frames_per_second` from `speed = np.sqrt(bvx**2 + bvy**2) * frames_per_second`. Only compare raw m/s against `surprise_speed_threshold`. |
| S5-M1 | **MEDIUM** | `detector.py:65` | `time_s = frame / 25.0` hardcoded instead of using `config.frames_per_second`. | Use `config.frames_per_second` for consistency. |
| S5-M2 | **MEDIUM** | `latency.py:100-107` | `frame_player_map` built via `df.iterrows()` — O(N) loop over every row. For large matches (500K+ rows) this is slow. | Use `df.groupby("frame_count")` or `pivot_table` instead. |
| S5-L1 | **LOW** | `latency.py:85` | Docstring says "heading angles (degrees)" but `compute_velocity_features` uses `np.arctan2` → **radians**. Code logic is correct (uses `np.degrees`), but docs mislead. | Update docstring: "heading (radians)". |
| S5-PASS | — | overall | Empty DataFrame handling, edge cases, and schema compliance all pass. Reaction times are non-negative and within bounds. | **PASS_WITH_FIXES** |

### Signal 3 — Pressing Accuracy (`src/signals/pressing/`)

| ID | Severity | File | Finding | Recommendation |
|----|----------|------|---------|----------------|
| S3-M1 | **MEDIUM** | `accuracy.py:115-122` | `aggregate_pressing_by_block` uses cross-join (`_key=1` merge) to assign frames to blocks. For N frames × B blocks, creates an N×B intermediate. With 100K frames × 20 blocks = 2M rows. | Use interval-based lookup (same pattern as drift's `aggregate_drift_by_block`). |
| S3-L1 | **LOW** | `detection.py:105` | `intercept_probability > 0.0` is very permissive — any tiny positive value counts. Could include noise in pressing detection. | Make threshold configurable or increase to > 0.05. |
| S3-L2 | **LOW** | `tti.py:25` | `compute_tta_threshold()` always uses defaults (15 m/s, 20 m). Not configurable at call site. | Accept optional parameters, defaulting to current values. |
| S3-PASS | — | overall | Schema validation passes. Accuracy values in [0,1]. Intercept probabilities bounded. | **PASS_WITH_FIXES** |

### Signal 1 — Positional Drift (`src/signals/positional_drift/`)

| ID | Severity | File | Finding | Recommendation |
|----|----------|------|---------|----------------|
| S1-H1 | **HIGH** | `bridge.py:298-306` | `_parse_v2_shapes` creates `ShapeWindow` objects with default `team_uuid=""`. `build_player_role_map` filters by `team_uuid`, so V2 shapes have no matching team → empty role map → Signal 1 produces no output. | Set `team_uuid` on V2 ShapeWindows. Requires parsing team UUID from the shape file structure (from `matchInfo.contestant` if shapes are per-team). |
| S1-M1 | **MEDIUM** | `drift.py` (all) | Uses `"frame"` column but loader produces `"frame_count"`. A workaround exists in `validate_all.py` but signals should handle this transparently. | Either (a) rename loader output to `frame`, or (b) add a `frame_col` parameter to drift functions defaulting to `"frame_count"`. |
| S1-M2 | **MEDIUM** | `aggregate_drift_by_block:150` | `inclusive="left"` in `between()` excludes block end frame. If blocks are [start, end] inclusive, this is an off-by-one. | Verify block boundary convention and match it. |
| S1-L1 | **LOW** | `bridge.py:224` | Fallback minute window calculation (0 for P1, 45 for P2) assumes no stoppage time. | Use `phase` × 45 min or estimate from actual elapsed time. |
| S1-PASS | — | overall | Drift values (when shape data is available) are positive, distance-based statistics are correct. Schema compliance good. | **PASS_WITH_FIXES** |

### Integration Fixes — Model 1 H-Fixes

| ID | Severity | File | Finding | Recommendation |
|----|----------|------|---------|----------------|
| INT-H1 | **PASS** | `composite.py` | `game_id` parameter added to `build_pressure_dataset` and propagated. Validated — works. | None needed. |
| INT-H2 | **PASS** | `composite.py` | Baseline computation correctly filters by `phase == 1` with graceful fallbacks. | None needed. |
| INT-H3 | **PASS** | `gk_utils.py`, `load_tracking.py` | Goalkeeper detection unified across all Model 1 modules. | None needed. |
| INT-L1 | **LOW** | `composite.py` | Indicator column names in `compute_pressure_composite` hardcoded (e.g., `"opponents_nearby_mean"`). New preprocessing stages with different column names silently produce zero contributions. | Document expected column names or make indicator configurable. |

### `validate_all.py`

| Issue | Verdict |
|-------|---------|
| Load and prepare | **PASS** |
| Model 1 (game_id, baseline, GK) | **PASS** |
| Signal 5 (reaction times) | **PASS** |
| Signal 3 (accuracy, schema) | **PASS** |
| Signal 1 (drift with coordinate fix) | **PASS** (for V1 shape files) |
| Results saved to JSON | **PASS** |

---

## Overall Verdict

**PASS_WITH_FIXES** ✅

Two HIGH issues were found and have been fixed (full Coder → Validator cycle). All tests pass with the fixes applied.

---

## Fixes Applied

### [S5-H1] FIXED ✅ — `detector.py:120`
**Ball speed over-multiplication removed.**

The `classify_transition_type` function was multiplying ball velocity by `frames_per_second` (25×), despite velocities already being in m/s (confirmed in `smoothing.py` where `np.gradient / 0.04dt` converts to m/s). This would have inflated ball speeds to 500 m/s at a 20 m/s pass, classifying nearly all transitions as "surprise".

**Fix**: Removed `* frames_per_second` from the speed computation. Surprise classification now correctly compares raw m/s against `surprise_speed_threshold` (10 m/s).

### [S1-H1] FIXED ✅ — `bridge.py:build_player_role_map`
**V2 shape file team UUID fallback added.**

The V2 shape parser (`_parse_v2_shapes`) creates `ShapeWindow` objects with `team_uuid=""` (default), causing `build_player_role_map` to produce empty role maps for V2 format shape files — no Signal 1 output.

**Fix**: Added fallback in `build_player_role_map`: when `team_uuid` filtering returns no matches, all windows for that minute are tried. Jersey-number matching naturally selects the correct team's roles. This also fixes edge cases where the team UUID in the shape file doesn't match any tracking team.

---

## Escalation to Conor

**No issues require Conor's decision.** Both HIGH issues had clear, non-controversial fixes. MEDIUM/LOW items can be tracked in the backlog.
