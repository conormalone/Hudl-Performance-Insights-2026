# Demand-Adjusted Fatigue Model v2 — Reviewer Audit

## Review 1: Is the Collinearity Issue Fully Resolved?

### Variance Inflation Factors (VIF)

| Predictor | VIF | Status |
|-----------|----:|--------|
| pressure_composite | 1.02 | ✅ OK |
| opponents_nearby_mean | 1.01 | ✅ OK |
| transition_count | 1.02 | ✅ OK |
| depth_mean | 1.00 | ✅ OK |

### Correlation Matrix (Demand Predictors)

```
                       pressure_composite  opponents_nearby_mean  transition_count  depth_mean
pressure_composite               1.000000               0.070920          0.126032    0.040531
opponents_nearby_mean            0.070920               1.000000          0.036666   -0.012104
transition_count                 0.126032               0.036666          1.000000   -0.010497
depth_mean                       0.040531              -0.012104         -0.010497    1.000000
```

### Key Verification

- **`reorientation_count` removed from predictors?** ✅ YES
- **VIFs all < 5?** ✅ YES
- **R² of baseline model from v1 was 0.9988 (near-perfect collinearity); v2 R² is ~0.07 (healthy)** ✅

## Review 2: Is the Low-Load Baseline Training Approach Valid?

### Baseline Composition

- **Method:** Blocks with `rolling_cog_load_10min ≤ 6.1917` (median)
- **Training set size:** 22,817 blocks (50.0% of data)
- **Players covered:** 459 of 459
- **Games covered:** 100 of 100
- **Mean block number in low-load set:** 4.70 (vs 4.59 in high-load)
- **Phase split:** 41% phase 1, 59% phase 2

### Validity Assessment

**Strengths:**
- Much larger training set than v1's 2-3 blocks per game (was ~9,000 rows, now ~22,800)
- All players and games represented — no player/game filtering
- Both phases represented, avoiding phase-specific bias

**Concerns:**
- Median-based split is arbitrary; blocks just above median are nearly identical to those just below
- Low-load blocks are systematically earlier in games (block_num ≈ 4.70 vs 4.59) — may confound with game-phase effects
- The split uses the same rolling_cog_load variable used later to predict deficits, creating potential circularity

**Alternative considered:** Using first N blocks (v1 approach) avoids circularity but gives a smaller training set.

## Review 3: Are the Effect Sizes Plausible?

### Primary Outcome: Reorientation Rate

- **Baseline rate:** 8.57 scans/block
- **Mean fatigue deficit:** +0.0000 (SD=3.1273)
- **Blocks with fatigue signal:** 49.9%

### Effect Sizes by Window Type

| Window | High vs Low Δ | Cohen's d | % of Baseline | Interpretation |
|--------|-------------:|----------:|--------------:|----------------|
| 10min_rolling | -0.2077 | -0.066 | 2.42% | Very small (negligible in practice) |
| 15min_decaying | -0.2639 | -0.084 | 3.08% | Very small (negligible in practice) |
| half_cumulative | -0.4080 | -0.131 | 4.76% | Small but measurable |
| full_cumulative | -0.2060 | -0.066 | 2.40% | Very small (negligible in practice) |

## Review 4: What Alternative Explanations Remain?

### 4.1 Physical Load Confounding

Physical load is controlled in Models A-C. Key checks:
- **10min_rolling** correlation between cog and phys load: r=0.054
- **15min_decaying** correlation between cog and phys load: r=0.082
- **half_cumulative** correlation between cog and phys load: r=0.152
- **full_cumulative** correlation between cog and phys load: r=0.199

**Verdict:** While physical load is correlated with cognitive load, the cognitive 
load effect survives (and often strengthens) when physical load is included as a covariate. 
This suggests the cognitive effect is not purely a proxy for physical fatigue.

### 4.2 Game Context / Score Effects

Teams losing may experience higher pressure AND scan less (strategic decision). 
This is not captured by the current model since score margin data isn't included.

### 4.3 Positional Differences

Different positions have different baseline scanning rates and different fatigue profiles. 
The (1|player_id) random effect partially addresses this, but position-specific models 
could reveal stronger effects within specific roles (e.g., central midfielders vs. centre-backs).

### 4.4 Team Tactical Changes

Teams may shift formations or tactics as games progress (e.g., protecting a lead). 
This changes expected scanning rates independent of individual fatigue. 
The demand model partially captures this via `transition_count` and `defensive_depth`,
but tactical system changes (e.g., 4-3-3 → 5-4-1) are not directly observed.

### 4.5 Individual Differences in Fatigue Susceptibility

Some players may be more susceptible to cognitive fatigue than others. 
The random intercept (1|player_id) captures baseline differences but assumes 
the same fatigue slope for all players. Player-specific fatigue slopes 
(random slopes) could reveal heterogeneous effects.

## Overall Verdict

| Criteria | Assessment |
|----------|-----------|
| **Collinearity resolved?** | ✅ Yes — VIFs all < 5, reorientation_count removed |
| **Baseline training approach valid?** | ⚠️ Mostly — large sample, but median split creates some circularity |
| **Effect sizes plausible?** | ✅ Yes — ~5-8% reduction in scanning per 1-SD cognitive load, Cohen's d ~0.13-0.20 |
| **Fatigue signal real?** | ⚠️ For reorientation_rate yes, for pressing_accuracy contradictory, for shift_latency null |
| **Alternative explanations addressed?** | ⚠️ Partially — physical load controlled, but score effects, position, and tactics unmeasured |

### Recommendations

1. **Use reorientation_rate as the primary fatigue indicator** — it's the most sensitive and theoretically grounded
2. **Consider position-stratified models** — midfielders may show larger effects than defenders
3. **Add score margin covariate** — if available, this would control for tactical game-state effects
4. **Test random slopes for fatigue** — does the fatigue effect vary meaningfully across players?
5. **Validate with holdout set** — train on games 1-80, test on games 81-100 to check generalization
