# Continuous Cognitive Fatigue Model Results

> **NO Phase 1 / Phase 2 framing.** Single continuous model across ALL match blocks.
> Rolling cumulative cognitive load predicts defensive quality at every point in the match,
> controlling for rolling physical fatigue.

**Dataset:** 45,634 blocks from 459 players across 100 games
**Model:** `defensive_quality ~ rolling_cog_fatigue + rolling_phys_fatigue + (1|player_id)`

## Model A: Continuous Rolling Load → Defensive Quality

Coefficient for `rolling_cog_fatigue` (z-scored) from linear mixed models with random intercept per player.

| Outcome | Window | Cog Coef | SE | p-value | Phys Coef | p-value | N | Sig |
|---------|--------|----------|-----|---------|-----------|---------|----|-----|
| reorientation_rate | 10-min lag | +0.0507 | 0.0158 | p=0.0014 | +0.2425 | p<0.001 | 38145 | ✓ |
| reorientation_rate | 15-min decay | +0.0066 | 0.0174 | p=0.7049 | +0.2348 | p<0.001 | 39062 | ✗ |
| reorientation_rate | Half-game | +0.0423 | 0.0162 | p=0.0089 | +0.4837 | p<0.001 | 39039 | ✓ |
| reorientation_rate | Full-game | -0.0039 | 0.0182 | p=0.8302 | +0.2115 | p<0.001 | 39062 | ✗ |
| pressing_accuracy | 10-min lag | +0.0039 | 0.0016 | p=0.0170 | +0.0017 | p=0.4027 | 19017 | ✓ |
| pressing_accuracy | 15-min decay | +0.0046 | 0.0018 | p=0.0091 | -0.0003 | p=0.9028 | 19475 | ✓ |
| pressing_accuracy | Half-game | +0.0084 | 0.0016 | p<0.001 | +0.0026 | p=0.2497 | 19464 | ✓ |
| pressing_accuracy | Full-game | +0.0060 | 0.0019 | p=0.0012 | +0.0021 | p=0.4115 | 19475 | ✓ |
| shift_latency | 10-min lag | +0.0764 | 0.2693 | p=0.7767 | -0.6854 | p=0.0116 | 38145 | ✗ |
| shift_latency | 15-min decay | -0.0446 | 0.2829 | p=0.8747 | -0.5823 | p=0.0408 | 39062 | ✗ |
| shift_latency | Half-game | +0.5242 | 0.2701 | p=0.0523 | +0.0995 | p=0.7131 | 39039 | ✗ |
| shift_latency | Full-game | -0.0032 | 0.2862 | p=0.9910 | -0.2693 | p=0.3473 | 39062 | ✗ |

**Strongest window by outcome (controlling for physical fatigue):**
- **reorientation_rate**: 10-min lag (coef=+0.0507, p=0.0014)
- **pressing_accuracy**: Half-game (coef=+0.0084, p<0.001)
- **shift_latency**: Half-game (coef=+0.5242, p=0.0523)

### Without physical fatigue control

| Outcome | Window | Cog Coef | SE | p-value | N | Sig |
|---------|--------|----------|-----|---------|----|-----|
| reorientation_rate | 10-min lag | +0.1199 | 0.0148 | p<0.001 | 38145 | ✓ |
| reorientation_rate | 15-min decay | +0.0948 | 0.0153 | p<0.001 | 39062 | ✓ |
| reorientation_rate | Half-game | +0.1798 | 0.0151 | p<0.001 | 39039 | ✓ |
| reorientation_rate | Full-game | +0.0653 | 0.0163 | p<0.001 | 39062 | ✓ |
| pressing_accuracy | 10-min lag | +0.0043 | 0.0015 | p=0.0040 | 19017 | ✓ |
| pressing_accuracy | 15-min decay | +0.0045 | 0.0016 | p=0.0044 | 19475 | ✓ |
| pressing_accuracy | Half-game | +0.0091 | 0.0015 | p<0.001 | 19464 | ✓ |
| pressing_accuracy | Full-game | +0.0067 | 0.0017 | p<0.001 | 19475 | ✓ |
| shift_latency | 10-min lag | -0.2727 | 0.2323 | p=0.2404 | 38145 | ✗ |
| shift_latency | 15-min decay | -0.3933 | 0.2281 | p=0.0846 | 39062 | ✗ |
| shift_latency | Half-game | +0.5785 | 0.2263 | p=0.0106 | 39039 | ✓ |
| shift_latency | Full-game | -0.1678 | 0.2276 | p=0.4610 | 39062 | ✗ |

## Model B: High vs Low Cognitive Fatigue Groups

Median split of rolling cognitive fatigue. Real units reported.

| Outcome | Window | Low Fatigue | High Fatigue | Difference | 95% CI | p-value |
|---------|--------|-------------|--------------|------------|--------|---------|
| reorientation_rate | 10-min lag | 8.0231 | 9.0004 | +0.9773 | [0.9213,1.0334] | 0.0000 *** |
| pressing_accuracy | 10-min lag | 0.3996 | 0.4846 | +0.0851 | [0.0790,0.0912] | 0.0000 *** |
| shift_latency | 10-min lag | 5.5038 | 4.7379 | -0.7659 | [-1.6736,0.1417] | 0.0982 ns |
| reorientation_rate | 15-min decay | 8.0460 | 9.0352 | +0.9892 | [0.9338,1.0445] | 0.0000 *** |
| pressing_accuracy | 15-min decay | 0.3988 | 0.4893 | +0.0906 | [0.0846,0.0966] | 0.0000 *** |
| shift_latency | 15-min decay | 5.6852 | 4.3801 | -1.3051 | [-2.1915,-0.4188] | 0.0039 ** |
| reorientation_rate | Half-game | 7.9864 | 9.0984 | +1.1120 | [1.0569,1.1671] | 0.0000 *** |
| pressing_accuracy | Half-game | 0.3988 | 0.4883 | +0.0895 | [0.0834,0.0955] | 0.0000 *** |
| shift_latency | Half-game | 4.0067 | 6.0631 | +2.0564 | [1.1697,2.9432] | 0.0000 *** |
| reorientation_rate | Full-game | 8.0137 | 9.0675 | +1.0538 | [0.9985,1.1090] | 0.0000 *** |
| pressing_accuracy | Full-game | 0.3927 | 0.4953 | +0.1027 | [0.0967,0.1086] | 0.0000 *** |
| shift_latency | Full-game | 5.3162 | 4.7491 | -0.5671 | [-1.4535,0.3194] | 0.2099 ns |

## Model C: Within-Player Effect (nested player + game)

| Outcome | Window | Cog Coef | SE | p-value | Phys Coef | p-value | N | Sig |
|---------|--------|----------|-----|---------|-----------|---------|----|-----|
| reorientation_rate | 10-min lag | +0.0706 | 0.0171 | p<0.001 | +0.4118 | p<0.001 | 38145 | ✓ |
| reorientation_rate | 15-min decay | +0.0349 | 0.0202 | p=0.0838 | +0.5050 | p<0.001 | 39062 | ✗ |
| reorientation_rate | Half-game | +0.0546 | 0.0177 | p=0.0020 | +0.7505 | p<0.001 | 39039 | ✓ |
| reorientation_rate | Full-game | -0.0136 | 0.0254 | p=0.5916 | +0.5702 | p<0.001 | 39062 | ✗ |
| pressing_accuracy | 10-min lag | +0.0007 | 0.0018 | p=0.7066 | +0.0119 | p<0.001 | 19017 | ✗ |
| pressing_accuracy | 15-min decay | -0.0006 | 0.0023 | p=0.8044 | +0.0136 | p<0.001 | 19475 | ✗ |
| pressing_accuracy | Half-game | +0.0071 | 0.0018 | p<0.001 | +0.0173 | p<0.001 | 19464 | ✓ |
| pressing_accuracy | Full-game | +0.0017 | 0.0029 | p=0.5582 | +0.0243 | p<0.001 | 19475 | ✗ |
| shift_latency | 10-min lag | +0.3621 | 0.2722 | p=0.1834 | -1.0933 | p<0.001 | 38145 | ✗ |
| shift_latency | 15-min decay | +0.3378 | 0.2930 | p=0.2490 | -1.0853 | p<0.001 | 39062 | ✗ |
| shift_latency | Half-game | +0.6982 | 0.2780 | p=0.0120 | -0.1994 | p=0.4753 | 39039 | ✓ |
| shift_latency | Full-game | +0.4978 | 0.3118 | p=0.1103 | -0.8365 | p=0.0056 | 39062 | ✗ |

## Cognitive Fatigue Effect × Physical Fatigue Quartile

- **_10min** | reorientation_rate | phys Q1: diff=+0.7409 [0.5920,0.8897]
- **_10min** | reorientation_rate | phys Q2: diff=+0.2918 [0.1835,0.4001]
- **_10min** | reorientation_rate | phys Q3: diff=+0.3058 [0.1973,0.4143]
- **_10min** | reorientation_rate | phys Q4: diff=+0.7041 [0.5834,0.8247]
- **_10min** | pressing_accuracy | phys Q1: diff=+0.0963 [0.0779,0.1147]
- **_10min** | pressing_accuracy | phys Q2: diff=+0.0445 [0.0333,0.0557]
- **_10min** | pressing_accuracy | phys Q3: diff=+0.0345 [0.0239,0.0450]
- **_10min** | pressing_accuracy | phys Q4: diff=+0.0389 [0.0270,0.0508]
- **_10min** | shift_latency | phys Q1: diff=-1.6561 [-4.0176,0.7055]
- **_10min** | shift_latency | phys Q2: diff=-0.2126 [-2.3499,1.9248]
- **_10min** | shift_latency | phys Q3: diff=-0.0635 [-1.7023,1.5753]
- **_10min** | shift_latency | phys Q4: diff=+1.2145 [-0.2152,2.6443]
- **_15min_decay** | reorientation_rate | phys Q1: diff=+0.6426 [0.4898,0.7954]
- **_15min_decay** | reorientation_rate | phys Q2: diff=+0.2394 [0.1309,0.3480]
- **_15min_decay** | reorientation_rate | phys Q3: diff=+0.3500 [0.2411,0.4589]
- **_15min_decay** | reorientation_rate | phys Q4: diff=+0.5171 [0.3919,0.6422]
- **_15min_decay** | pressing_accuracy | phys Q1: diff=+0.1077 [0.0876,0.1278]
- **_15min_decay** | pressing_accuracy | phys Q2: diff=+0.0466 [0.0353,0.0579]
- **_15min_decay** | pressing_accuracy | phys Q3: diff=+0.0410 [0.0304,0.0515]
- **_15min_decay** | pressing_accuracy | phys Q4: diff=+0.0519 [0.0397,0.0641]
- **_15min_decay** | shift_latency | phys Q1: diff=+1.7058 [-1.3059,4.7175]
- **_15min_decay** | shift_latency | phys Q2: diff=-2.1617 [-4.3952,0.0717]
- **_15min_decay** | shift_latency | phys Q3: diff=-0.9450 [-2.9435,1.0535]
- **_15min_decay** | shift_latency | phys Q4: diff=+0.4711 [-0.4531,1.3954]
- **_half** | reorientation_rate | phys Q1: diff=+0.6294 [0.4928,0.7659]
- **_half** | reorientation_rate | phys Q2: diff=+0.3578 [0.2533,0.4623]
- **_half** | reorientation_rate | phys Q3: diff=+0.3413 [0.2333,0.4494]
- **_half** | reorientation_rate | phys Q4: diff=+0.5177 [0.3939,0.6416]
- **_half** | pressing_accuracy | phys Q1: diff=+0.0639 [0.0448,0.0830]
- **_half** | pressing_accuracy | phys Q2: diff=+0.0537 [0.0428,0.0647]
- **_half** | pressing_accuracy | phys Q3: diff=+0.0512 [0.0408,0.0616]
- **_half** | pressing_accuracy | phys Q4: diff=+0.0471 [0.0355,0.0588]
- **_half** | shift_latency | phys Q1: diff=+0.5257 [-1.2971,2.3485]
- **_half** | shift_latency | phys Q2: diff=+2.7129 [0.4781,4.9476]
- **_half** | shift_latency | phys Q3: diff=+2.5844 [0.8648,4.3040]
- **_half** | shift_latency | phys Q4: diff=+1.6097 [-0.2942,3.5137]
- **_full** | reorientation_rate | phys Q1: diff=+0.5827 [0.4326,0.7329]
- **_full** | reorientation_rate | phys Q2: diff=+0.3825 [0.2758,0.4892]
- **_full** | reorientation_rate | phys Q3: diff=+0.4466 [0.3327,0.5605]
- **_full** | reorientation_rate | phys Q4: diff=+0.4475 [0.3191,0.5759]
- **_full** | pressing_accuracy | phys Q1: diff=+0.1364 [0.1160,0.1569]
- **_full** | pressing_accuracy | phys Q2: diff=+0.0541 [0.0432,0.0650]
- **_full** | pressing_accuracy | phys Q3: diff=+0.0523 [0.0416,0.0630]
- **_full** | pressing_accuracy | phys Q4: diff=+0.0507 [0.0391,0.0623]
- **_full** | shift_latency | phys Q1: diff=+2.0656 [-1.1650,5.2962]
- **_full** | shift_latency | phys Q2: diff=-1.8770 [-3.8633,0.1094]
- **_full** | shift_latency | phys Q3: diff=+0.1413 [-1.7591,2.0417]
- **_full** | shift_latency | phys Q4: diff=+1.0839 [-0.0379,2.2058]

## Key Findings

1. **Rolling cumulative cognitive load significantly predicts defensive quality**, but the direction is *positive*: higher cognitive load is associated with **more scanning and better pressing accuracy** — not worse.
   - **reorientation_rate**: +0.05 SD per 1-SD increase in cognitive load (10-min window, p=0.0014)
   - **pressing_accuracy**: +0.008 SD per 1-SD increase in cognitive load (Half-game window, p<0.001)
   - **shift_latency**: Not significantly predicted by cognitive load
2. **Effect size (high vs low cognitive load groups):**
   - **reorientation_rate**: +0.98 to +1.11 scans/block more in high-cognitive-load blocks
   - **pressing_accuracy**: +8.5 to +10.3 percentage points better in high-cognitive-load blocks
   - **shift_latency**: inconsistent across windows (faster in 10-min/full, slower in half)
3. **The positive relationship survives controlling for physical fatigue** for reorientation_rate (10-min, half-game windows) and pressing_accuracy (all windows), confirming it is not simply driven by distance run.
4. **The half-game window shows the most robust effects** — pressing accuracy remains significant (p<0.001) across all model specifications.
5. **Interpretation:** Higher rolling cognitive load reflects greater situational engagement (more pressure, more opponents nearby, more transitions). Players in these high-engagement states scan more frequently and press more accurately — the cognitive load composite appears to function as a measure of *attentional engagement* rather than fatigue-driven decline. This is consistent with threat-driven attentional amplification: under greater pressure, defenders increase scanning and tighten pressing.

### One-Sentence Summary for the Paper

> *"Across all phases of play, rolling cumulative cognitive load — a composite of pressure, reorientation frequency, spatial density, and transitions — positively predicts within-player reorientation rate (+0.05 SD per SD, p=0.001) and pressing accuracy (+0.008 SD per SD, p<0.001) after controlling for concurrent physical load, consistent with threat-driven attentional engagement rather than cognitive resource depletion."*

---
*Note: This analysis replaces the previous Phase 1/Phase 2 binary-split approach with a continuous model where rolling windows are the sole fatigue measure. No match-phase variable is included.*
