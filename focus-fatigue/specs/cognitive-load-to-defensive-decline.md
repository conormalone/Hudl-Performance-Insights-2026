# Cognitive Load → Cumulative Fatigue → Defensive Decline

## The Feedback Loop (NO Phase 1/Phase 2 framing)

```
Every block in the match:
                              ┌─── Rolling cognitive fatigue ──┐
                              │   (cumulative pressure,        │
                              │    reorientations, space)      │
                              │                                ▼
Cognitive load indicators ────┤                    Defensive quality at block t
(per-block)                    │                    (reorientation_rate,
                              │                     pressing accuracy,
                              ├─── Rolling physical fatigue ──┤ shift latency)
                              │   (cumulative distance, HSR)   │
                              └───────────────────────────────────────┘
```

**The core model: at any block t, does rolling accumulated cognitive fatigue predict defensive quality in that block, controlling for rolling physical fatigue?**

No binary split. No Phase 1 vs Phase 2. The rolling windows are the fatigue measure.

## Data

The parquet has per-block data for each (player, game, block_num). Block-level signals are already calculated: `reorientation_rate`, `pressing_accuracy`, `shift_latency`, `positional_drift`, `physical_load`.

Cognitive load indicators (per-block):
- `pressure_composite` — how much pressure the defender was under
- `opponents_nearby_mean` — spatial pressure / how crowded it was
- `reorientation_count` / `transition_count` — how engaged they were
- `depth_mean` — how much space they had to cover

Defensive quality outcomes:
- `reorientation_rate` — scans per frame (cognitive quality)
- `pressing_accuracy` — successful press rate (technical quality)
- `shift_latency` — time to react to ball movement (reaction speed)
- `positional_drift` — body shape discipline

## How it works (no Phase 1/Phase 2)

For each (player, game), for each block position t (t=0, 1, 2, ...):
1. Compute rolling cumulative fatigue UP TO block t using only PRECEDING blocks (t-1, t-2, ...)
   - **10-min rolling**: mean of last 2 preceding blocks (5 min each)
   - **15-min decaying**: exponentially weighted mean of all preceding blocks, tau=15
   - **Half-game**: mean of all blocks in the same half up to t
   - **Full-game**: mean of all preceding blocks from start of match
2. For EACH of cognitive load AND physical load (separate rolling measures)
3. `rolling_cog_fatigue_t` = composite of rolling pressure + rolling reorientations + rolling opponents_nearby
4. `rolling_phys_fatigue_t` = rolling mean of physical_load up to t
5. Outcome = defensive quality at block t

## Model (single, applies across ALL blocks equally)

`defensive_quality_at_t ~ rolling_cog_fatigue_at_t + rolling_phys_fatigue_at_t + (1|player_id) + (1|game_id)`

Run for each:
- Outcome variable: reorientation_rate, pressing_accuracy, shift_latency, positional_drift
- Rolling window type: 10-min, 15-min decay, half, full
- Also run without rolling_phys_fatigue to see what happens when physical load IS included

## High vs Low Cognitive Fatigue Groups (the headline comparison)

1. For each (player, game, block), compute rolling_cog_fatigue
2. Split all blocks at the median → "high cognitive fatigue" vs "low cognitive fatigue"
3. Compare defensive quality between groups:
   - "Cognitively fatigued players scan X fewer times per block [CI]"
   - "Cognitively fatigued players press X percentage points worse [CI]"
4. ALSO compare controlling for rolling_phys_fatigue (to ensure the difference isn't just tired legs)

## Key Question

> *"At any point in a match, does a player with higher accumulated cognitive fatigue show worse defensive quality than a player with lower accumulated cognitive fatigue, controlling for how much they've run?"*

## Expected Findings

| Outcome | Expected | Why |
|---------|----------|-----|
| Reorientation rate | Strong | It's the pure cognitive signal |
| Pressing accuracy | Attenuated | Confounded with physical state |
| Shift latency | Moderate | Reaction time slows with fatigue |
| Positional drift | Weak/None | Confounded with tactics |

## Output
Write results to `focus-fatigue/outputs/analysis/continuous_cog_load_model.md`
Figure to `focus-fatigue/outputs/analysis/continuous_cog_load_figure.png`
Push to origin/main.
