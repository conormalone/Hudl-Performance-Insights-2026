# Performance-Context Confidence Intervals

Comparison of **reorientation_rate**, **shift_latency**, and **pressing_accuracy** between Phase 1 (first half) and Phase 2 (second half).

---
## 1. Raw Means with 95% CI

| Metric | Phase 1 Mean [95% CI] | Phase 2 Mean [95% CI] | Δ (P2 − P1) [95% CI] | n₁ | n₂ |
|--------|----------------------|----------------------|----------------------|----|----|
| Reorientation Rate | 9.2283 [9.1862, 9.2704] | 7.9454 [7.9056, 7.9851] | -1.2829 [-1.3408, -1.2250] | 22155 | 23479 |
| Shift Latency | 9.6232 [8.6794, 10.5670] | 1.3519 [1.3479, 1.3559] | -8.2713 [-9.2151, -7.3275] | 21626 | 22550 |
| Pressing Accuracy | 0.4421 [0.4379, 0.4464] | 0.4325 [0.4284, 0.4367] | -0.0096 [-0.0155, -0.0037] | 11058 | 11702 |

---
## 2. Real-World Units (per 5-min Block)

### Reorientation Rate → Scans per 5-min Block

- **Phase 1:** 46.14 scans/block [45.93, 46.35]
- **Phase 2:** 39.73 scans/block [39.53, 39.93]
- **Drop:** -6.41 scans/block [-6.70, -6.13]

*(Conversion: reorientation_rate (per min) × 5 min = scans per 5-min block)*

### Shift Latency → Extra Seconds per Shift

- **Phase 1:** 9.6232 s [8.6794, 10.5670]
- **Phase 2:** 1.3519 s [1.3479, 1.3559]
- **Δ (P2 − P1):** -8.2713 s [-9.2151, -7.3275]

### Pressing Accuracy → Percentage Points per Block

- **Phase 1:** 44.21% [43.79%, 44.64%]
- **Phase 2:** 43.25% [42.84%, 43.67%]
- **Δ (P2 − P1):** -0.96 pp [-1.55, -0.37]

---
## 3. Per-Half Aggregates (~9 blocks/half)

*(Assuming ~9 × 5-min blocks per half)*

### Reorientation — Total Scans Lost per Half
- Drop per block: -6.41 scans [-6.70, -6.13]
- **Total lost per half:** -57.73 scans [-60.34, -55.13]

### Shift Latency — Total Extra Reaction Time per Half
- Extra per shift: -8.2713 s [-9.2151, -7.3275]
- Avg transition count per block: 0.986
- Avg shifts per half: 8.88
- **Total extra reaction time per half:** -73.41 s [-81.79, -65.04]

### Pressing Accuracy — Percentage-Point Drop per Half
- Drop per block: -0.96 pp [-1.55, -0.37]
- **Total accuracy deficit per half:** -8.65 pp [-13.96, -3.33]  (cumulative across 9 blocks)

---
## 4. Effect Sizes (Cohen's d) with 95% CI

| Metric | Cohen's d | 95% CI | Interpretation |
|--------|-----------|--------|----------------|
| Reorientation Rate | -0.4071 | [-0.4257, -0.3886] | small |
| Shift Latency | -0.1669 | [-0.1856, -0.1483] | negligible |
| Pressing Accuracy | -0.0423 | [-0.0683, -0.0163] | negligible |

---
## 5. Paper-Ready Results Table

| Metric | Phase 1 | Phase 2 | Δ (95% CI) | d (95% CI) | Per-half impact |
|--------|---------|---------|------------|------------|----------------|
| Reorientation (scans/block) | 46.1 | 39.7 | -6.4 [-6.7, -6.1] | -0.41 [-0.43, -0.39] | −58 scans/half |
| Shift latency (s) | 9.62 | 1.35 | -8.27 [-9.22, -7.33] | -0.17 [-0.19, -0.15] | -73 s/half |
| Pressing accuracy (pp) | 44.2% | 43.3% | -0.96 [-1.55, -0.37] | -0.04 [-0.07, -0.02] | -8.6 pp total |

---
## 6. Summary Narrative

**Reorientation rate** drops from 46.1 to 39.7 scans per 5-min block (Δ = -6.4 [-6.7, -6.1]), representing a loss of approximately **58 scans in the second half**. The effect size is 0.41 [-0.43, -0.39] (medium).

**Shift latency** changes from 9.62 s to 1.35 s (Δ = -8.27 s [-9.22, -7.33]), yielding a total of **-73 seconds of reduced latency per half** (effect size d = -0.17 [-0.19, -0.15]). The direction is negative, indicating faster reactions in Phase 2 — consistent with a game-tempo effect rather than fatigue-driven slowing.

**Pressing accuracy** declines from 44.2% to 43.3% (Δ = -0.96 pp [-1.55, -0.37]), representing a cumulative **-8.6 percentage-point accuracy deficit across the second half** (effect size d = -0.04 [-0.07, -0.02], negligible).

