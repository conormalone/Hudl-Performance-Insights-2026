# Defensive Group Fatigue Analysis

**Analysis date:** 2026-07-18

## Methodology

### Position Clusters

Players were clustered by behavioral averages and labeled as CB, FB, DM, or CM/W.
Distribution: {'CB': np.int64(104), 'CM/W': np.int64(140), 'DM': np.int64(8), 'FB': np.int64(146)}

### Defensive Group

- **Defensive group (CB + FB + DM):** 258 players (27,279 blocks)
- **Non-defensive (CM/W):** 201 players (18,355 blocks)

### Key Design Choices

1. **Rolling load from PRECEDING blocks only** — no future information leaks
2. **10-min rolling window**: mean of 2 preceding blocks (~10 min)
3. **15-min decaying window**: exponentially weighted mean of ALL preceding blocks (τ=15 min)
4. **Composite cognitive load z-score**: standardized average of 5 components (pressure_composite, opponents_nearby_mean, reorientation_count, transition_count, depth_mean)
5. **75th/25th percentile thresholds**: high load ≥ 75th, low load ≤ 25th, middle 50% discarded
6. **Demand-adjusted model**: predict expected reorientation_rate from CURRENT situation (pressure_composite + opponents_nearby_mean + depth_mean) — no reorientation_count in predictors
7. **Low-load baseline**: lowest 50% of rolling cognitive load used to train demand model
8. **Physical load controlled** in every model

## Demand Model Quality

### Defenders

- **R²** = 0.0694 (variance explained by situational factors in low-load baseline)
- **Training set**: 13,640 low-load blocks
- **Predictors**: pressure_composite, opponents_nearby_mean, depth_mean
- ✅ **R² ≥ 0.05** — demand model has meaningful explanatory power.

| Predictor | β (std) | p-value |
|----------|--------:|--------:|
| Intercept | +8.3376 | 0.0000 *** |
| pressure_composite_z | +0.8445 | 0.0000 *** |
| opponents_nearby_mean_z | +0.0298 | 0.2597  |
| depth_mean_z | -0.1216 | 0.0000 *** |

### All Players

- **R²** = 0.0308 (variance explained by situational factors in low-load baseline)
- **Training set**: 22,817 low-load blocks
- **Predictors**: pressure_composite, opponents_nearby_mean, depth_mean
- ⚠️ **R² < 0.05** — demand model has limited explanatory power. Interpret with caution.

| Predictor | β (std) | p-value |
|----------|--------:|--------:|
| Intercept | +8.6528 | 0.0000 *** |
| pressure_composite_z | +0.4863 | 0.0000 *** |
| opponents_nearby_mean_z | +0.3282 | 0.0000 *** |
| depth_mean_z | -0.0311 | 0.1532  |

## High vs Low Cognitive Load: Fatigue Deficit

Format: more negative deficit = worse than situationally expected = fatigue signal.

### Defenders

| Window | N(high) | N(low) | Mean(high) | Mean(low) | Diff | 95% CI | p-value | Cohen's d | Survives Phys Ctrl |
|--------|-------:|------:|----------:|---------:|-----:|-------|--------:|----------:|:------------------:|
| 10-min Rolling | 6820 | 6820 | -0.3378 | -0.5222 | +0.1843 | [0.0279, 0.3317] | 0.0161 | 0.041 | ❌ No |
| 15-min Decay | 6820 | 6820 | -0.3553 | -0.4747 | +0.1195 | [-0.0385, 0.2711] | 0.1246 | 0.026 | ❌ No |

### All Players

| Window | N(high) | N(low) | Mean(high) | Mean(low) | Diff | 95% CI | p-value | Cohen's d | Survives Phys Ctrl |
|--------|-------:|------:|----------:|---------:|-----:|-------|--------:|----------:|:------------------:|
| 10-min Rolling | 11409 | 11409 | +0.2687 | -0.7725 | +1.0412 | [0.9419, 1.1380] | 0.0000 | 0.274 | ✅ Yes |
| 15-min Decay | 11409 | 11409 | +0.3070 | -0.8065 | +1.1134 | [1.0151, 1.2093] | 0.0000 | 0.296 | ✅ Yes |

## Continuous Model: deficit ~ rolling_cog_z + rolling_phys_z

### Defenders

| Window | β(cog) | p(cog) | β(phys) | p(phys) |
|--------|-------:|-------:|--------:|--------:|
| 10-min Rolling | -1.2227 | 0.0000 | — | — |
| 15-min Decay | -1.4232 | 0.0000 | — | — |

### All Players

| Window | β(cog) | p(cog) | β(phys) | p(phys) |
|--------|-------:|-------:|--------:|--------:|
| 10-min Rolling | -0.8507 | 0.0000 | — | — |
| 15-min Decay | -1.0846 | 0.0000 | — | — |

## Key Findings

### 1. Defenders Only (10-min Rolling)

- **High cognitive load deficit:** -0.3378 scans/block
- **Low cognitive load deficit:** -0.5222 scans/block
- **Difference:** +0.1843 [95% CI: 0.0279, 0.3317]
- **p-value:** 0.016070
- **Cohen's d:** 0.041
- **Survives physical load control:** ❌ No (β=+0.0125, p=0.8993)
- ⚠️ **Reversed effect:** Defenders show MORE scanning under high cognitive load (possible compensation/arousal effect).

### 2. All Players (10-min Rolling)

- **High cognitive load deficit:** +0.2687 scans/block
- **Low cognitive load deficit:** -0.7725 scans/block
- **Difference:** +1.0412 [95% CI: 0.9419, 1.1380]
- **p-value:** 0.000000
- **Survives physical load control:** ✅ Yes

### 3. Consistency Check
- Both groups show a **positive** deficit direction — consistent across defensive and all players.

### 4. Methodological Notes

- **Demand model R² (defenders):** 0.0694 ✅
- **Demand model R² (all players):** 0.0308 ⚠️
- **Demand predictors exclude reorientation_count** to avoid collinearity
- **Rolling load uses PRECEDING blocks only** — no look-ahead bias
- **Position labels from behavioral clustering** (not pitch coordinates)
