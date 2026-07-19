# Rolling Window Fatigue Methodology

## Problem
Physical load and cognitive signals are currently computed per static 5-minute block. This means both measure the same time slice, making dissociation impossible — any correlation between them is partly structural.

## Solution
Recompute BOTH physical load AND cognitive signals on identical rolling/decaying windows. Then test whether cognitive fatigue effects survive controlling for physical fatigue on the same window.

## Window Types (all four, applied identically to physical and cognitive)

### 1. 10-min Rolling (lagging)
- For each block at position `t`, take blocks `[t-1, t]` (2 blocks = 10 min, assuming 5-min blocks)
- **Physical:** Mean `physical_load` across those 2 blocks
- **Cognitive:** Mean signal value across those 2 blocks
- Edge case: block `t=0` (first block) → use only block 0

### 2. 15-min Decaying (exponential)
- For each block at position `t`, compute exponentially weighted average:
  - Weight for block at lag `i`: `w_i = exp(-i * 5 / tau)` where tau = 15 min decay constant
  - Normalise weights to sum to 1
- **Physical:** Weighted mean of `physical_load` over all preceding blocks
- **Cognitive:** Weighted mean of signal over all preceding blocks
- This captures accumulated fatigue with recency bias

### 3. Half-game (45 min)
- Phase 1 blocks → first-half load: mean `physical_load` across all phase 1 blocks
- Phase 2 blocks → second-half load: mean `physical_load` across all phase 2 blocks
- **Physical:** Per-half mean physical load
- **Cognitive:** Per-half mean signal value
- This captures half-level accumulation

### 4. Full-game (90 min)
- One value per player per game: mean `physical_load` across ALL blocks
- **Physical:** Per-game mean physical load
- **Cognitive:** Per-game mean signal value
- This captures whole-match accumulation

## Data Structure
The parquet already has per-block data. The rolling windows can be computed per (player_id, game_id) group, sorted by `block_num`.

## Dissociation Analysis (for each window type)

For each cognitive signal × window type combination:

### Model 1: Univariate
`signal_windowed ~ phase`
- Does the signal show a Phase 1 → Phase 2 decline?
- Report: coefficient, p-value, Cohen's d

### Model 2: Controlled
`signal_windowed ~ phase + phys_load_windowed`
- Does the Phase effect survive controlling for physical load on the same window?
- Report: phase coefficient, p-value, physical load coefficient, p-value
- Key metric: Δ in phase coefficient from Model 1 to Model 2

### Model 3: Interaction
`signal_windowed ~ phase * phys_load_windowed`
- Does the cognitive decline depend on physical load level?
- Report: interaction term, p-value, simple slopes

## Expected Findings (hypotheses)

| Signal | Window | Expected Result |
|--------|--------|----------------|
| Reorientation Rate | 10-min / 15-min decay | Survives physical control (genuine cognitive) |
| Reorientation Rate | Half / Full | Also survives (accumulated cognitive load) |
| Pressing Accuracy | 10-min / 15-min decay | Attenuated or eliminated (tracking immediate physical state) |
| Pressing Accuracy | Half / Full | Flat or reversed (not a cumulative fatigue signal) |
| Shift Latency | 10-min / 15-min decay | Survives (reaction time is cognitive) |
| Positional Drift | 10-min / 15-min decay | Probably confounded (physical tiredness → body shape drift) |
| Transition Latency | All | Weak/no signal (not a fatigue-sensitive metric) |

## Output
- `outputs/analysis/rolling_window_results.md` — full report with tables for each window type
- `outputs/analysis/rolling_window_figure.png` — key visualisation
- Update analysis notebook with code
