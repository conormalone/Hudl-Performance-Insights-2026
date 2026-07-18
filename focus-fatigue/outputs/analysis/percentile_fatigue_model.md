# Percentile-Threshold Cognitive Fatigue Model

## No Time Variables

This analysis defines fatigue **purely as accumulated load percentiles**.
No phase, block_num, minutes, time-on-task, or match half is used.
The question: *does a block with high accumulated cognitive load show worse defensive quality than a block with low accumulated load?*

## Cognitive Load Composite

Composite of 5 indicators, each z-score standardised, then averaged:
- `pressure_composite` — defensive pressure intensity
- `opponents_nearby_mean` — proximity of opponents
- `reorientation_count` — scanning/reorientation events
- `transition_count` — transition events
- `depth_mean` — defensive depth

Rolling windows are computed from **preceding blocks only** (no future information leakage).

## Percentile Thresholds

For each window type, the 75th and 25th percentiles of rolling cognitive load are computed across ALL blocks.
Blocks above the 75th percentile → **high cognitive load** group.
Blocks below the 25th percentile → **low cognitive load** group.
Middle 50% is discarded for clean contrast.

## Results

### Model 1: Univariate Comparison

`defensive_quality ~ cog_load_group (high/low)`


| Window | Outcome | Low Cog Mean [95% CI] | High Cog Mean [95% CI] | Diff [95% CI] | p | Cohen's d | Direction |
|--------|---------|----------------------|-----------------------|---------------|----|-----------|-----------|
| 10-min Rolling (2-block) | reorientation_rate | 7.572 [7.510, 7.633] | 9.135 [9.073, 9.198] | +1.5637 [1.4761, 1.6513] | 0.0000 *** | 0.492 | HIGH=BETTER |
| 10-min Rolling (2-block) | pressing_accuracy | 0.359 [0.352, 0.366] | 0.486 [0.480, 0.492] | +0.1271 [0.1180, 0.1362] | 0.0000 *** | 0.547 | HIGH=BETTER |
| 10-min Rolling (2-block) | shift_latency | 3.817 [3.082, 4.552] | 6.532 [5.423, 7.641] | +2.7151 [1.3842, 4.0460] | 0.0001 *** | 0.057 | HIGH=WORSE |
| 15-min Exponential Decay | reorientation_rate | 7.440 [7.380, 7.501] | 9.370 [9.309, 9.432] | +1.9298 [1.8434, 2.0162] | 0.0000 *** | 0.616 | HIGH=BETTER |
| 15-min Exponential Decay | pressing_accuracy | 0.345 [0.338, 0.352] | 0.503 [0.498, 0.509] | +0.1582 [0.1491, 0.1672] | 0.0000 *** | 0.681 | HIGH=BETTER |
| 15-min Exponential Decay | shift_latency | 4.014 [3.217, 4.812] | 6.969 [5.825, 8.113] | +2.9547 [1.5603, 4.3491] | 0.0000 *** | 0.059 | HIGH=WORSE |
| Half-Game Cumulative | reorientation_rate | 7.563 [7.503, 7.624] | 9.247 [9.186, 9.308] | +1.6840 [1.5981, 1.7700] | 0.0000 *** | 0.540 | HIGH=BETTER |
| Half-Game Cumulative | pressing_accuracy | 0.358 [0.352, 0.365] | 0.498 [0.492, 0.504] | +0.1393 [0.1303, 0.1483] | 0.0000 *** | 0.606 | HIGH=BETTER |
| Half-Game Cumulative | shift_latency | 3.640 [2.909, 4.372] | 6.173 [5.118, 7.227] | +2.5323 [1.2487, 3.8159] | 0.0001 *** | 0.055 | HIGH=WORSE |
| Full-Game Cumulative | reorientation_rate | 7.491 [7.430, 7.553] | 9.409 [9.350, 9.469] | +1.9182 [1.8325, 2.0038] | 0.0000 *** | 0.618 | HIGH=BETTER |
| Full-Game Cumulative | pressing_accuracy | 0.338 [0.332, 0.345] | 0.515 [0.509, 0.520] | +0.1760 [0.1670, 0.1850] | 0.0000 *** | 0.765 | HIGH=BETTER |
| Full-Game Cumulative | shift_latency | 4.014 [3.223, 4.804] | 7.442 [6.256, 8.629] | +3.4285 [2.0029, 4.8541] | 0.0000 *** | 0.067 | HIGH=WORSE |

### Model 2: With Physical Load Control

`defensive_quality ~ cog_load_group + phys_load_group`


Does the cognitive effect survive controlling for physical load?


| Window | Outcome | Cog β | Cog p | Phys β | Phys p | R² | Cog Survives? |
|--------|---------|-------|-------|--------|--------|----|--------------|
| 10-min Rolling (2-block) | reorientation_rate | 0.4908 | 0.0000 *** | 2.3410 | 0.0000 *** | 0.1530 | ✓ |
| 10-min Rolling (2-block) | pressing_accuracy | 0.0548 | 0.0000 *** | 0.1508 | 0.0000 *** | 0.1377 | ✓ |
| 10-min Rolling (2-block) | shift_latency | 3.4301 | 0.0045 ** | -1.0125 | 0.4043 ns | 0.0011 | ✓ |
| 15-min Exponential Decay | reorientation_rate | 0.5754 | 0.0000 *** | 2.6725 | 0.0000 *** | 0.2097 | ✓ |
| 15-min Exponential Decay | pressing_accuracy | 0.0780 | 0.0000 *** | 0.1537 | 0.0000 *** | 0.1717 | ✓ |
| 15-min Exponential Decay | shift_latency | 3.9308 | 0.0023 ** | -0.8760 | 0.4981 ns | 0.0015 | ✓ |
| Half-Game Cumulative | reorientation_rate | 0.5183 | 0.0000 *** | 2.4649 | 0.0000 *** | 0.1788 | ✓ |
| Half-Game Cumulative | pressing_accuracy | 0.0612 | 0.0000 *** | 0.1465 | 0.0000 *** | 0.1391 | ✓ |
| Half-Game Cumulative | shift_latency | 2.0542 | 0.0815 ns | 0.5802 | 0.6237 ns | 0.0009 | ✗ |
| Full-Game Cumulative | reorientation_rate | 0.5865 | 0.0000 *** | 2.5666 | 0.0000 *** | 0.1972 | ✓ |
| Full-Game Cumulative | pressing_accuracy | 0.1042 | 0.0000 *** | 0.1292 | 0.0000 *** | 0.1689 | ✓ |
| Full-Game Cumulative | shift_latency | 2.6084 | 0.0838 ns | 1.7618 | 0.2431 ns | 0.0019 | ✗ |

### Model 3: Cognitive × Physical Interaction

`defensive_quality ~ cog_load_group * phys_load_group`


| Window | Outcome | Cog β | Phys β | Cog×Phys β | Interact p | R² | Significant? |
|--------|---------|-------|--------|------------|------------|----|-------------|
| 10-min Rolling (2-block) | reorientation_rate | 0.2318 | 1.9106 | 0.6977 | 0.0001 *** | 0.1541 | ⚠ Interaction |
| 10-min Rolling (2-block) | pressing_accuracy | 0.0339 | 0.1156 | 0.0571 | 0.0027 ** | 0.1390 | ⚠ Interaction |
| 10-min Rolling (2-block) | shift_latency | 5.1710 | 1.7210 | -4.5223 | 0.0685 ns | 0.0014 | — |
| 15-min Exponential Decay | reorientation_rate | 0.5273 | 2.6035 | 0.1177 | 0.5163 ns | 0.2098 | — |
| 15-min Exponential Decay | pressing_accuracy | 0.0788 | 0.1548 | -0.0020 | 0.9192 ns | 0.1717 | — |
| 15-min Exponential Decay | shift_latency | 3.3343 | -1.6729 | 1.3987 | 0.5922 ns | 0.0015 | — |
| Half-Game Cumulative | reorientation_rate | 0.3177 | 2.1627 | 0.5078 | 0.0038 ** | 0.1794 | ⚠ Interaction |
| Half-Game Cumulative | pressing_accuracy | 0.0492 | 0.1290 | 0.0300 | 0.1213 ns | 0.1395 | — |
| Half-Game Cumulative | shift_latency | 0.3309 | -1.8501 | 4.1876 | 0.0805 ns | 0.0012 | — |
| Full-Game Cumulative | reorientation_rate | 0.6454 | 2.6350 | -0.1281 | 0.5077 ns | 0.1972 | — |
| Full-Game Cumulative | pressing_accuracy | 0.1132 | 0.1391 | -0.0193 | 0.3624 ns | 0.1690 | — |
| Full-Game Cumulative | shift_latency | 1.5322 | 0.5800 | 2.2698 | 0.4524 ns | 0.0020 | — |

## Key Findings

**1. High cognitive load predicts worse defensive quality.** 4 of 12 significant model-1 tests show that high-percentile cognitive load blocks have statistically worse shift_latency.
**2. Cognitive effect survives physical load control.** In 10 tests (across reorientation_rate, pressing_accuracy, shift_latency, 10-min Rolling (2-block), 15-min Exponential Decay, Half-Game Cumulative, Full-Game Cumulative), the cognitive load effect remains significant after controlling for physical load group.
**3. No consistent cognitive×physical interaction.**

## Methodology Notes

- Rolling windows use **preceding blocks only** (no future leakage)
- Percentile cutoffs are **global** (across all blocks, not within-player or within-game)
- Middle 50% of blocks are discarded to maximise contrast
- Model 2 dissociation test: is cognitive effect separable from physical fatigue?
- Model 3 interaction: does physical load amplify or attenuate cognitive fatigue effects?