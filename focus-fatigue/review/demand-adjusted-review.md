# Reviewer Critique: Demand-Adjusted Fatigue Model

## Independent Data Science Audit

*This review evaluates the demand-adjustment methodology, baseline validity, control adequacy, and alternative explanations.*

## 1. Does the Demand-Adjustment Actually Isolate Fatigue from Demand?

### Conceptual Soundness

The core idea is theoretically sound: by regressing out the effect of current situational factors (pressure, opponent proximity, transition count, etc.), the residual captures deviation from situation-expected behaviour. If fatigue causes players to underperform relative to what the situation demands, this is precisely the signal we want.

✅ **Strengths:**
- The demand variables chosen (pressure_composite, opponents_nearby_mean, reorientation_count, transition_count, depth_mean) are the right set of situational factors that should predict defensive scanning behaviour.
- The model explains 58.9% of variance in reorientation rate (R²=0.589), confirming that situational factors are strong predictors of scanning behaviour — a necessary condition for the residual approach to work.
- Pressure composite alone is nearly uncorrelated with reorientation after controlling for the other variables (β=+0.0001, p=0.93), suggesting its contribution is mediated through more granular situational factors.

### Methodological Concerns

⚠️ **Demand variables are not independent of fatigue.**
The reorientation_count variable (number of reorientations in this block) appears on BOTH sides of the equation: it's used to predict reorientation_rate. Since reorientation_rate = reorientation_count / normalized_frames, this creates a near-deterministic relationship. The extremely high R² (99.88%) in the baseline model (first 2-3 blocks) is a red flag — it suggests near-perfect collinearity between a predictor and the outcome, making the residuals essentially measurement noise rather than a fatigue signal.

⚠️ **The well-rested baseline model shows this clearly.**
  - R² = 0.9988 — essentially all variance is explained by the demand variables alone
  - reorientation_count coefficient β=+0.0087 (p<0.001) dominates
  - Fatigue deficits from this model are nearly all ~0, which is why the sensitivity analysis shows no significant cognitive load effects
  - This is because reorientation_count and reorientation_rate are deterministically linked (rate = count / frame_normalization)

⚠️ **The full-model approach (Approach A) avoids this collinearity problem** because the large sample gives more reliable estimates, but reorientation_count still dominates. The coefficient for reorientation_count (β=+0.0066) drives most of the prediction, while the other demand variables contribute modestly.

**Recommendation:** Remove reorientation_count from the demand model. It creates an artifactual relationship. Use only pressure_composite, opponents_nearby_mean, transition_count, and depth_mean — factors that describe the situation without being a direct count of the behaviour being predicted.

## 2. Is the 'Well-Rested' Baseline Valid?

### What Was Done
The 'well-rested' baseline was estimated by fitting the demand model on the first 2-3 blocks of each game per player, under the assumption that fatigue hasn't accumulated yet in these early blocks.

✅ **Strengths:**
- Using early-game blocks is a natural approach to capturing the demand-response relationship in a relatively unfatigued state.
- The first 2-3 blocks are early enough (first ~15 min of game time) to be before substantial fatigue accumulation for most players.

❌ **Critical Problems:**

1. **Near-perfect fit (R²=0.9988) is impossible for a real fatigue model.**
   - This R² value means 99.88% of variance in reorientation_rate is 'explained' by demand variables alone on early blocks.
   - For comparison, the full-model R² is 0.589 — more reasonable.
   - The discrepancy suggests the baseline model is overfitted, almost certainly because reorientation_count almost perfectly determines reorientation_rate (rate = count / constant_frames).

2. **First blocks may not be 'well-rested'.**
   - Players may arrive with pre-existing fatigue from travel, training, or previous matches.
   - Some players enter games late (substitutes) and their 'first blocks' are later in game time.
   - The first blocks of each game include the warm-up-to-competition transition, which has unique demand characteristics.

3. **Within-player baseline insufficient.**
   - Only 2-3 blocks per player per game gives at most 3 data points for estimating each player's demand-response relationship.
   - These few observations are insufficient for reliable individual-level prediction.
   - A better approach: estimate the demand-response relationship across ALL low-accumulated-load blocks (e.g., blocks where preceding load is in the bottom quartile).

**Recommendation:** Use a cross-validated estimator of the demand-reorientation relationship trained on all blocks where preceding accumulated cognitive load is low (bottom quartile within each player-game), not just the first 2-3 blocks. This gives more data per player and better generalisation.

## 3. Are the Controls Sufficient?

### Physical Load Control
✅ Physical load is controlled in the main models (`fatigue_deficit ~ cog_load + phys_load`).
✅ The cognitive load effect survives physical load control on ALL window types (p < 0.001).
✅ All high vs low comparisons also survive physical load control.

### Missing Controls

⚠️ **Game context.** The model does not control for:
- Score state (winning/losing/drawing) — affects risk-taking and engagement
- Time of match — late-game effects beyond accumulated load
- Opponent quality — stronger opponents may induce different scanning patterns
- Substitution status — fresh substitutes have different fatigue profiles
- Venue (home/away) — travel fatigue and home advantage

⚠️ **Individual differences.** No player-level random effects.
- Some players are naturally high-scanning (central defenders) vs low-scanning (forwards)
- A mixed model with random slopes for demand variables would separate within-player fatigue from between-player trait differences
- Currently, the 63.3% deficit-negative rate mostly reflects that the model's residuals are symmetric around zero, not that most players are fatigued.

⚠️ **Cluster dependence.** Blocks within the same player-game are correlated. Standard errors are narrower than they should be. Cluster-robust SEs or mixed effects are needed.

**Recommendation:** At minimum, add random intercepts for player and game. If feasible, include score state and opponent quality as controls.

## 4. What Alternative Explanations Remain?

### 4a. Reverse Causality
Players who are inherently more engaged (higher scanning baseline) may:
- Generate more reorientations in general (higher reorientation_count)
- End up in situations with higher pressure_composite (because scanning surfaces more threats)
- Have higher accumulated load BECAUSE they're more engaged, not because load impairs them

This is especially problematic given collinearity between reorientation_count and reorientation_rate.

### 4b. Confounded Situations
High accumulated cognitive load may simply mean the player was in consecutive high-demand situations. The 'demand-adjustment' removes the direct effect of current demand variables, but cannot remove the effect of the SITUATION TYPE (e.g., transition phases are fundamentally different from set-piece phases in ways not captured by the demand variables).

### 4c. Physical Load Mediation
Physical load has a STRONG positive effect on fatigue deficit (β = +0.15 to +0.21, p < 0.001 on all windows). Higher physical load predicts MORE positive deficits (players scan MORE than expected after high physical exertion). This is the opposite of the fatigue hypothesis and could indicate:
- Arousal: physical exertion increases alertness short-term
- Recovery: blocks with high preceding physical load may be followed by lower-intensity play where scanning catches up
- Correlation with engagement phases (high physical load occurs during intense play when all players are more engaged)

### 4d. Hypervigilance Interpretation
The analysis finds ~36.7% of blocks have positive deficits (scanning MORE than expected). These could be:
- Genuine hypervigilance (fatigue-induced compensatory effort, as documented in sleep deprivation research)
- Appropriate engagement that the demand model failed to capture (missing demand variables)
- Measurement artefact (reorientation_count variance not fully explained)

### 4e. Effect Magnitude Is Small
The largest high-vs-low deficit difference is -0.1374 (half_cumulative window).
For a player averaging 8.6 scans per block over ~5 minutes, this is ~0.14 fewer scans than expected — about 1.6% reduction.
Even under physical load control, the effect grows to β=-0.1867 (about 2.2% reduction).
This is statistically significant but may not be practically meaningful for match outcomes.

## 5. Overall Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Conceptual soundness | ★★★★☆ | Strong idea — residualising demand is the right approach to isolate fatigue |
| Methodology execution | ★★★☆☆ | Solid but has a critical collinearity issue with reorientation_count |
| Baseline validity | ★★☆☆☆ | Well-rested baseline model is near-perfect due to deterministic relationship; needs fixing |
| Controls | ★★★☆☆ | Physical load controlled well; missing context variables and random effects |
| Alternative explanations | ★★★★☆ | Well-discussed but some (reverse causality, hypervigilance) need deeper exploration |
| Practical significance | ★★★☆☆ | Effect is ~0.14 fewer scans/block — statistically robust but small in real terms |

### Key Fixes Needed

1. **Remove reorientation_count from the demand model.** It creates a deterministic relationship with reorientation_rate. Use only: pressure_composite, opponents_nearby_mean, transition_count, depth_mean.
2. **Use player-level random effects** to separate within-player fatigue from between-player trait differences.
3. **Add block-clustered standard errors** to account for within-game dependence.
4. **Compute demand model on low-accumulated-load blocks** (bottom quartile) rather than just the first 2-3 blocks to get more robust baseline estimates.
5. **Include game context controls** (score state, opponent quality, phase type) to reduce residual confounding.

### Verdict

The demand-adjusted approach is **directionally correct** and represents a meaningful improvement over raw composite comparison. The finding that accumulated cognitive load predicts more negative deficits after controlling for physical load is robust across all window types (p < 0.001). However, the **reorientation_count collinearity** undermines the well-rested baseline model and likely inflates the full-model R². With this fix applied, the signal would likely weaken but should remain directionally consistent given the consistent pattern across window types.

Despite these methodological caveats, the **cross-window consistency** (all 4 windows show significant negative cognitive load effects, all survive physical control) suggests a real but small fatigue effect: players with high accumulated cognitive load scan ~0.1-0.2 fewer times per block than the situation demands.

---
*Review generated by demand-adjusted fatigue model reviewer.*