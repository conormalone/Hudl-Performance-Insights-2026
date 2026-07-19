# Demand-Adjusted Fatigue Model

## Problem
The raw composite measures situational demand, not fatigue. Players under more pressure scan MORE, not less — they're doing their job. Comparing high vs low load groups will always show "high load = better performance" because players are appropriately engaged.

## Solution: Expected vs Actual Performance

For each block, compute what a well-rested player SHOULD do given the current situation. The deviation from expected is the fatigue signal.

### Step 1: Model Expected Performance

Fit a model predicting defensive quality from CURRENT situational factors only (no accumulated load):
```
reorientation_rate ~ pressure_composite + opponents_nearby_mean + reorientation_count + transition_count + depth_mean
```

Do this ONLY on early blocks where fatigue hasn't accumulated (first 2-3 blocks of each game per player) to establish a "well-rested baseline" relationship between demand and output.

Or simpler: fit the model across ALL data, then residual = what's left unexplained by current demand.

### Step 2: Compute Fatigue Deficit

```
fatigue_deficit = actual_reorientation_rate - predicted_reorientation_rate
```

Negative deficit = player scanning less than the situation demands (FATIGUE)
Zero deficit = player doing what's expected given the demand (HEALTHY)
Positive deficit = player scanning more than expected (UNUSUAL — hypervigilance?)

### Step 3: Test Against Accumulated Load

Does accumulated cognitive load (rolling windows from preceding blocks) predict more negative deficits?

```
fatigue_deficit ~ rolling_cog_load + rolling_phys_load
```

Then split into high/low rolling_cog_load groups (75th percentile) and compare mean deficit.

### Step 4: Real Units

"How many fewer scans per block" should a player have made given the situation?
Mean fatigue_deficit in units of reorientation_rate, with CI.

## Outcome
- `outputs/analysis/demand_adjusted_fatigue_model.md`
- `outputs/analysis/demand_adjusted_fatigue_figure.png`
- `focus-fatigue/review/demand-adjusted-review.md` — reviewer critique
- Push to origin/main
