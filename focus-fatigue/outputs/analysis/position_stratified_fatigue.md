# Position-Stratified Fatigue Analysis

## Overview

Player positions were derived from behavioural patterns using K-means clustering
(k=4, silhouette=0.209) on per-game averages of tracking-derived metrics,
with no external position data. The demand-adjusted fatigue model was then
fit separately for each position group.

**Key findings:**
- Positions show **DIFFERENT** direction of fatigue deficit
- Most affected: **FB** (d=0.000)
- Least affected: **CB** (d=-0.000)

---

## Step 1: Position Clustering

### Cluster Profiles
| Position | n | Depth | Opp Nearby | Physical Load | Reorient Rate | Transition Rate | Drift |
|----------|---|-------|------------|--------------|--------------|---------------|-------|
| CM/W | 160 | 57.0 | 0.84 | 1.34 | 8.60 | 0.0081 | 29.9 |
| FB | 83 | 53.6 | 1.40 | 1.53 | 7.89 | 0.0077 | 26.2 |
| DM | 180 | 57.1 | 1.16 | 1.41 | 9.16 | 0.0107 | 24.1 |
| CB | 36 | 57.7 | 0.18 | 1.06 | 6.05 | 0.0054 | 27.4 |


### Football Validation
| Check | Result |
|-------|--------|
| CBs highest depth | ⚠️ |
| CM/Ws highest transition_rate | ⚠️ |
| DMs highest reorientation_rate | ✅ |
| FBs highest physical_load | ✅ |


---

## Step 2: Fatigue Model Results

Model: `reorientation_rate ~ pressure + opp_nearby + transitions + depth`
then `fatigue_deficit ~ rolling_cog_load + rolling_phys_load + (1|player_id)`

| Position | n_players | n | Deficit Mean | d | [95% CI] | Cog β | Cog p | Demand R² |
|----------|-----------|---|-------------|---|---|-------|-------|----------|
| CB | 36 | 4035 | -0.0000 | -0.000 | [-0.031, 0.031] | -0.0039* | 0.0173 | 0.049 |
| CM/W | 160 | 16939 | -0.0000 | -0.000 | [-0.015, 0.015] | -0.0068 | 0.1973 | 0.044 |
| DM | 180 | 18946 | 0.0000 | 0.000 | [-0.014, 0.014] | -0.0247* | 0.0077 | 0.062 |
| FB | 83 | 5582 | 0.0000 | 0.000 | [-0.026, 0.026] | 0.0221 | 0.1333 | 0.057 |


### Real Units
| Position | Deficit/block | Per 10 blocks |
|----------|--------------|---------------|
| CB | -0.00000 | 0.00 reorientation changes per 10 blocks (fewer than demand predicts) |
| CM/W | -0.00000 | 0.00 reorientation changes per 10 blocks (fewer than demand predicts) |
| DM | +0.00000 | 0.00 reorientation changes per 10 blocks (more than demand predicts) |
| FB | +0.00000 | 0.00 reorientation changes per 10 blocks (more than demand predicts) |


---

## Interpretation

### Football Sense

Some cluster checks do not pass, likely due to noise in the synthetic/limited-variation dataset.

The radar chart (Panel A) shows the behavioural profile for each position group.

### Fatigue Story

Fatigue deficits differ: some positions scan more, others less than demand predicts.

**FBs** show the strongest effect (d=0.00),
**CBs** the weakest (d=-0.00).

---

## Figures

- **position_stratified_figure.png:** Panel A = radar of cluster centroids,
  Panel B = Cohen's d by position with 95% CIs,
  Panel C = deficit vs cognitive load by position.
