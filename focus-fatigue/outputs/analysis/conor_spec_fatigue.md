# Conor's Spec: Defensive Group Percentile Fatigue

## Methodology

### Position Assignment
Players were clustered into position groups using per-game averages of depth_mean, opponents_nearby_mean, physical_load, and reorientation_rate. Groups: CB, FB, DM, CM/W. Defensive = CB + FB + DM.

### Rolling Cognitive Load
Computed from **preceding blocks only** (10-min rolling = mean of preceding 2 blocks). Composite = mean of z-scored: pressure_composite, opponents_nearby_mean, reorientation_count, transition_count, depth_mean.

### Exclusion
First 2 blocks per (player, game) are excluded — these have no preceding blocks, so rolling load is zero.

### Load Groups
- **High load**: blocks ≥ 75th percentile of rolling cognitive load
- **Low load**: blocks ≤ 25th percentile of rolling cognitive load
- Middle 50% discarded

### Fatigue Deficit
Expected reorientation_rate modeled from pressure_composite + opponents_nearby_mean + depth_mean (NO reorientation_count). Model trained on blocks below median rolling cognitive load. Deficit = actual − expected.

### Physical Load Control
Physical load rolling mean (preceding 2 blocks, z-scored). High/low groups defined by same percentile method. Comparison repeated within each physical load stratum.

---

## Results

### Fatigue Model (Training: blocks below median rolling cognitive load)
- Features: pressure_composite, opponents_nearby_mean, depth_mean
- Coefficients: pressure_composite=0.016442, opponents_nearby_mean=0.830904, depth_mean=-0.004617
- Intercept: 7.306104
- R² on training data: 0.0288
- Training n: 19762

### 1. Defenders (CB + FB + DM)

| Group | n | Mean Deficit | 95% CI |
|-------|---|-------------|--------|
| High cognitive load | 7422 | 0.8961 | [0.8212, 0.9711] |
| Low cognitive load | 5651 | -0.8281 | [-0.9105, -0.7458] |

**Difference (low − high):** -1.7242 [-1.8356, -1.6129]
**Welch t-test:** t = 30.3467, p = 0.000000

**Physical load controlled:**

- Within **low physical load**: High load mean = -1.2764 (n=534), Low load mean = -1.1140 (n=4326), Diff = 0.1624, t = -0.7036, p = 0.481958

- Within **high physical load**: High load mean = 1.5494 (n=3562), Low load mean = 0.2913 (n=118), Diff = -1.2581, t = 5.0009, p = 0.000002

**Survives physical load control:** NO

---

### 2. All Players (no position filter)

| Group | n | Mean Deficit | 95% CI |
|-------|---|-------------|--------|
| High cognitive load | 9881 | 0.9720 | [0.9060, 1.0380] |
| Low cognitive load | 9881 | -0.3953 | [-0.4585, -0.3320] |

**Difference (low − high):** -1.3673 [-1.4587, -1.2759]
**Welch t-test:** t = 29.3156, p = 0.000000

**Physical load controlled:**

- Within **low physical load**: High load mean = -1.1415 (n=702), Low load mean = -0.8096 (n=6024), Diff = 0.3319, t = -1.5920, p = 0.111791

- Within **high physical load**: High load mean = 1.5833 (n=4573), Low load mean = 0.5699 (n=412), Diff = -1.0134, t = 7.4894, p = 0.000000

**Survives physical load control:** NO

---

## Summary

### Defenders (CB + FB + DM)
- Mean fatigue deficit was **-0.8281 at low cognitive load** and **0.8961 at high cognitive load**
- Difference (low − high): **-1.7242** [-1.8356, -1.6129]
- **p = 0.000000** — statistically significant at α = 0.05
- Survives physical load control: **NO**

### All Players
- Mean fatigue deficit was **-0.3953 at low cognitive load** and **0.9720 at high cognitive load**
- Difference (low − high): **-1.3673** [-1.4587, -1.2759]
- **p = 0.000000** — statistically significant at α = 0.05
- Survives physical load control: **NO**
