# Reviewer Task: Defensive Group Fatigue Analysis

## Context
A defensive group fatigue analysis has been completed. The analysis script is at `focus-fatigue/analysis/defensive_group_fatigue.py`. The report is at `focus-fatigue/outputs/analysis/defensive_group_fatigue.md`. The figure is at `focus-fatigue/outputs/analysis/defensive_group_fatigue.png`.

The data file is at `focus-fatigue/outputs/analysis/unified_fatigue_dataset.parquet` (in the repo root at `/home/conormalone/Hudl-Performance-Insights-2026/`).

## What to Review

### 1. Demand Model Validity (R² check)
Is the demand model valid? 
- R² (defenders): 0.0694
- R² (all players): 0.0308
- Check: is R² > 0.02? If not, the "expected" performance is unreliable.

### 2. Effect Direction Check
The results show a **paradox**:
- **Continuous model**: More cognitive load → more NEGATIVE deficit (correct fatigue direction) - β ≈ -1.22, p < 0.001
- **Percentile split (high vs low)**: High load group has LESS negative deficit than low load group (+0.18 diff, p=0.016)

This is the OPPOSITE of what fatigue would predict. The high cognitive load group is performing BETTER than the low cognitive load group (relative to situational expectations), even though the continuous model shows the right direction.

**Hypothesis**: The relationship may be non-linear. The middle 50% (discarded in percentile split) may contain the most negative deficits.

### 3. Methodology Cleanliness
Check:
- ✓ Preceding blocks only for rolling load (not current or future)
- ✓ No Phase 1/Phase 2 framing
- ✓ No time variables (block_num, minutes, halves)
- ✓ Percentile thresholds for load groups (75th = high, 25th = low)
- ✓ Demand model excludes reorientation_count (no collinearity)
- ✓ Physical load as control

### 4. Specific Questions
1. Is the demand model's 3-predictor set appropriate? (pressure_composite + opponents_nearby_mean + depth_mean)
2. Should transition_count be included in the demand model?
3. Is the percentile split vs continuous model disagreement a real finding or an artifact?
4. Does the analysis correctly handle the "no position mapping" players (2925 rows)?

## Output
Write your review to `focus-fatigue/review/defensive-group-review.md`
Include:
- Verdict on each question
- Verdict on demand model validity
- Verdict on effect directions
- Verdict on methodology cleanliness
- An overall pass/fail recommendation with explanation

Report back to me (Jervis) with your findings.
