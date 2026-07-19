# Clean Percentile Fatigue Analysis — Reviewer Task

## Files to Review

1. `focus-fatigue/analysis/clean_percentile_fatigue.py` — analysis script
2. `focus-fatigue/outputs/analysis/clean_percentile_fatigue.md` — generated report
3. `focus-fatigue/outputs/analysis/clean_percentile_fatigue_summary.json` — summary data

## Methodology to Verify

1. **Rolling load computation**: Are preceding-blocks-only rolling windows computed correctly? Any look-ahead bias?
2. **Contamination removal**: Are first 2 blocks per player-game correctly excluded? Check the `is_contaminated` logic.
3. **Defensive group filtering**: Only CB, FB, DM positions included?
4. **Demand model**: Does it predict reorientation_rate from pressure_composite, opponents_nearby_mean, depth_mean (no reorientation_count)?
5. **Continuous model**: Does `fatigue_deficit ~ rolling_cog_load_z + rolling_phys_load_z` return valid coefficients?
6. **Percentile split**: Are 25th/75th percentiles computed from the clean (post-removal) subset?
7. **Mixed model divergence**: OLS shows NO cognitive effect, but mixed model (player RE) shows NEGATIVE effect. Verify this is correct and not an artifact.

## Checks

- Run the script from scratch to confirm reproducibility
- Spot-check rolling load values for a few random (player, game) pairs
- Verify `fatigue_deficit` computation: deficit = actual - predicted
- Confirm that removing first blocks actually changes the percentile thresholds
- Check for any data leakage (future information used in predictions)

## Report

Output a brief report as `focus-fatigue/outputs/analysis/percentile_reviewer_audit.md` summarizing findings.
