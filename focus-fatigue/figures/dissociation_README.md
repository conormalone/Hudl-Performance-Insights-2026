# Dissociation Figure — Small Multiples Refactor

## Summary
Faceted scatter plot demonstrating cognitive fatigue effects on defensive quality are *independent* of physical load.

## Current Data State (2026-07-19)
| Metric | Value |
|---|---|
| Matches | 1 (2215790) |
| Players | 22 (11 per team) |
| Blocks per player | 1 (block 0 = first 5 min) |
| Signals with data | shift_latency (n=4), pressing_accuracy (n=10) |
| Signals empty | positional_drift, transition_latency |

## Figure Layout
- **Rows**: One per signal (shift_latency, pressing_accuracy)
- **Columns**: Physical load tertile (Low / High — only 2 due to binary pressure_composite values)
- **X-axis**: `reorientation_rate` (cognitive load proxy — scans/min)
- **Y-axis**: Raw signal value (deficit mode available when >3 blocks/player)

## Key Methodological Decisions

1. **Cognitive Load Metric**: `reorientation_rate` — measures scanning frequency. Higher = more situation-awareness checks = higher cognitive/attentional demand. Best available proxy in Model 1 outputs.

2. **Physical Load**: `pressure_composite` — composite of opponents_nearby, depth_mean, reorientation_count. Split via binary cut at 4.5 (only values 4.0 and 5.0 present).

3. **Baseline**: Per-player baseline from first 3 blocks. Currently falls back to raw values since only 1 block/player.

4. **No signal deficits**: With 1 block/player, baseline = block value → deficits = 0. Code is structured to auto-detect this and use raw values.

## Current Correlations (limited, n=1 block)
| Signal | Overall r | p-value | n |
|---|---|---|---|
| shift_latency vs reorientation_rate | -0.785 | 0.215 | 4 |
| pressing_accuracy vs reorientation_rate | 0.325 | 0.359 | 10 |

## Output Files
- `figures/dissociation.png` (300 DPI)
- `figures/dissociation.svg` (vector)
- `analysis/dissociation_figure.py` (reproducible pipeline)

## To Run
```bash
cd /mnt/usb/project/focus-fatigue
python3 src/merge_outputs.py
python3 analysis/dissociation_figure.py --raw-signals
```

## When More Data Arrives
1. Multiple blocks per player → per-player baselines from first 3 blocks → signal deficits
2. Wider pressure_composite range → proper tertile split (3 columns)
3. positional_drift & transition_latency → full 4-row figure
4. Run without `--raw-signals` to use deficit mode
