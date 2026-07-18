# Cognitive Load → Cumulative Fatigue → Defensive Decline

## The Feedback Loop

```
Phase 1                                Phase 2
─────────                              ─────────
Cognitive load ──────────────────────┐  Defensive quality
(pressure, reorientations, space)    │  (reorientation rate,
                                     │   pressing accuracy,
Physical load ─────┐                 │   shift latency)
(distance, HSR)    │                 │
                   ▼                 ▼
           Rolling fatigue measures   Predictive model:
           (10-min, 15-min decay,    rolling_fatigue ~ next_block_quality
            half, full)              controlling for physical load
```

## Data

The parquet has per-block data for each (player, game, block_num). Block-level signals are already calculated: `reorientation_rate`, `pressing_accuracy`, `shift_latency`, `positional_drift`, `transition_latency`, `physical_load`.

Phase 1 cognitive load indicators:
- `pressure_composite` — how much pressure the defender was under
- `opponents_nearby_mean` — spatial pressure / how crowded it was
- `reorientation_count` / `transition_count` — how engaged they were
- `depth_mean` — how much space they had to cover

Phase 2 defensive quality outcomes:
- `reorientation_rate` — scans per frame (cognitive quality)
- `pressing_accuracy` — successful press rate (technical quality)
- `shift_latency` — time to react to ball movement (reaction speed)

## Models (run for each rolling window type × outcome combination)

### Step 1: Split data by Phase
- Phase 1 blocks (t=0..~9): measure cognitive load
- Phase 2 blocks (t=~10..~18): measure defensive quality

### Step 2: Compute rolling fatigue measures
For each window type, compute a cumulative fatigue score for each player at each Phase 2 block position:
- **10-min rolling**: mean of last 2 blocks
- **15-min decaying**: exponentially weighted mean, tau=15
- **Half-game**: Phase 1 mean (applies to all Phase 2 blocks)
- **Full-game**: running mean from start of match

Compute for BOTH:
- `rolling_cog_load` — from pressure_composite, reorientation_count, etc.
- `rolling_phys_load` — from physical_load

### Step 3: Predictive models

**Model A — Simple predictor (Phase 1 → Phase 2)**
`outcome_Phase2 ~ cog_load_Phase1 + phys_load_Phase1`
- Does cognitive load in Phase 1 predict defensive quality in Phase 2?
- Key: is cog_load coefficient negative and significant?
- Does it remain significant with phys_load in the model?

**Model B — Rolling predictor (lagged)**
`outcome_at_block_t ~ rolling_cog_load_at_block_t + rolling_phys_load_at_block_t`
- Does accumulated cognitive fatigue predict defensive quality at each Phase 2 block?
- Key: negative coefficient for rolling_cog_load

**Model C — High vs low groups**
Split players into high/low cognitive fatigue groups (median split on rolling_cog_load). Compare Phase 2 defensive quality between groups.
- Do high-cog-fatigue players show worse defensive quality?
- How big is the difference (scans lost, presses missed)?

**Model D — Fixed effects**
`outcome ~ phase + rolling_cog_load + rolling_phys_load + (1|player_id) + (1|game_id)`
- Mixed model controlling for player and game
- Does rolling_cog_load predict within-player decline in Phase 2?

## Expected Findings

| Signal | Expected | Why |
|--------|----------|-----|
| Reorientation rate | Strongest predictor | Purely cognitive — drops when brain is tired |
| Pressing accuracy | Weak/confounded | Tracks physical state more than cognitive |
| Shift latency | Moderate | Reaction time should slow with cognitive load |
| Positional drift | Weak | Confounded with both physical and tactical |

## Output
Write results to `focus-fatigue/outputs/analysis/cog_load_to_defensive_decline.md`
Figure to `focus-fatigue/outputs/analysis/cog_load_vs_defensive.png`
Push to origin/main.
