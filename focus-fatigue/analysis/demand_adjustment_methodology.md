# Demand-Adjusted Pressing Accuracy — Methodology Investigation

**Project:** Focus Fatigue — Cognitive Fatigue Detection in Football Defence
**Author:** PM sub-agent audit (Jervis → Conor review)
**Date:** 19 July 2026
**Status:** Investigation complete — document for review

---

## Executive Summary

"Demand-adjusted pressing accuracy" was briefly floated as a way to control for
the difficulty of pressing situations a defender faces. This document investigates
whether the current data and pipeline support it.

**Bottom line:** A quick proxy is feasible (compute `intercept_probability` mean
over pressing-only frames instead of all frames), but a *proper* demand-adjusted
model requires event-level interception outcomes, a trained expected-value model,
and richer feature engineering — none of which exist in the current pipeline.
Recommendation: **abandon** the name "demand-adjusted" (it implies calibration we
don't have), but optionally add a **pressing difficulty covariate** as a control
variable.

---

## 1. Current Pipeline — What It Actually Computes

### 1.1 Data Flow

```
tracking.parquet (25fps, xy coords)
    ↓ load_tracking_statsperform()     renames + DOP-normalises
    ↓ smooth_trajectory()              Savitzky-Golay (window=7, poly=2)
    ↓ compute_velocity_features()      adds vx_smooth, vy_smooth, v_mag, heading
    ↓ compute_tti()                    Bekkers TTI per defender-attacker pair
    ↓ detect_pressing_events()         flags frames where defender is pressing
    ↓ classify_pressing_accuracy()     flags correct vs wasteful presses
    ↓ aggregate_pressing_by_block()    per-block, per-player aggregation
```

### 1.2 The TTI Model (Bekkers Time-To-Intercept)

At each frame, for each defender–attacker pair (≤30m apart):

```
tta_threshold = pass_distance / pass_speed = 20.0 / 15.0 ≈ 1.33s      (*hardcoded*)

def_speed = max(√(vx² + vy²), 0.1)
tau_dist  = distance / def_speed
cos_θ     = (def_v·dx + def_vy·dy) / (def_speed × distance)
tau_β     = β × (1 - cos_θ) × tau_dist                                    β=1.0
tti_value = reaction_time + tau_dist + tau_β                               t_reaction=0.2s

intercept_probability = σ(-k × (tta_threshold − tti_value))                k=3.0
```

This is a **physics-based heuristic** — not a calibrated probability.
Key hardcoded parameters (pass_speed=15 m/s, pass_distance=20 m, k=3.0,
reaction_time=0.2s) would need to be tuned for the specific league/competition.

### 1.3 Pressing Detection

A defender is "pressing" when all three conditions hold in a frame:

| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| Speed | ≥ 2.0 m/s | Moving fast enough to be actively closing down |
| Angle to attacker | ≤ 45° | Moving roughly toward the attacker (not running away) |
| Intercept probability | > 0.0 | Non-zero chance of reaching the attacker's position |

### 1.4 Accuracy Classification

A press is classified as **"correct"** if `intercept_probability > 0.18`.

The 0.18 threshold is arbitrary — not derived from empirical calibration.

### 1.5 Aggregation (Per Block, Per Player)

```python
grouped.agg(
    n_presses = is_pressing.sum(),
    correct_presses = is_correct_press.sum(),
    pressing_accuracy = correct_presses / n_presses,
    mean_intercept_prob = intercept_probability.mean(),        # ← ALL frames
    p90_tti = tti_value.quantile(0.90),
)
```

**Critical issue:** `mean_intercept_prob` is averaged over ALL frames in the
block, not just pressing frames. This severely dilutes the statistic because
non-pressing frames (typically 70–85% of frames) have much lower intercept
probabilities (mean ≈ 0.055 vs 0.192 for pressing frames in our sample).

---

## 2. Can We Do Demand-Adjustment with Current Data?

### 2.1 The Good News: Per-Frame Data IS Preserved

Running the pipeline on 100,000 frames (~7 match-minutes, all defenders):

| Metric | Pressing frames | Non-pressing frames |
|--------|----------------|---------------------|
| N frames | 6,789 | 39,168 |
| Mean `intercept_probability` | **0.192** | 0.055 |
| Median `intercept_probability` | **0.015** | 0.000 |
| % of total frames | 14.8% | 85.2% |

The `intercept_probability` column exists in the per-frame dataframe passed to
`aggregate_pressing_by_block()` — so we could compute a separate aggregation:

```python
# What we could add (one extra aggregation line):
mean_intercept_prob_pressing = intercept_probability[is_pressing].mean()
```

This would give, for each block and player, the **average physics-model intercept
probability of the pressing actions they actually attempted**.

### 2.2 Possible Quick-Proxies (and Why They're Weak)

**Proxy 1: Pressing Difficulty Ratio**
```
pressing_difficulty_ratio = pressing_accuracy / mean_intercept_prob_pressing
```
If a defender achieves 0.60 accuracy on presses averaging 0.20 intercept prob,
their ratio is 3.0 — "3× better than expected."

**Proxy 2: Pressing Difficulty Delta**
```
pressing_difficulty_delta = pressing_accuracy − mean_intercept_prob_pressing
```
Positive = better than the model would predict.

**Problem:** Both use `intercept_probability` as if it were a calibrated expected
value — **it is not**. Consider:

1. **Hardcoded parameters.** `pass_speed=15 m/s` is a world-class driven pass.
   A lofted through-ball or a slow square pass would have a very different
   intercept time, but the model applies the same threshold to all.

2. **Arbitrary threshold.** The 0.18 "correct press" cutoff and the 0.0 "is
   pressing" cutoff are not empirically validated.

3. **No outcome data.** The model never sees whether a press actually succeeded
   (interception) or failed (attacker kept possession). It only uses kinematics.

4. **Defender-centric only.** The model ignores the attacker's skill, body
   orientation, whether they're looking to pass vs. dribble, etc.

5. **Skewed distribution.** The per-pressing-frame `intercept_probability` has
   median 0.015 but mean 0.192 — driven by a long tail of high-probability
   events. This means most presses have near-zero predicted intercept probability,
   making the mean a poor summary.

### 2.3 Verdict on Current-Data Demand Adjustment

| Aspect | Feasible? | Detail |
|--------|-----------|--------|
| Compute pressing-only intercept prob | ✅ Yes | ~5 lines added to aggregation |
| Use as "expected accuracy" | ❌ No | Physics heuristic, not calibrated |
| Use as "pressing difficulty covariate" | ⚠️ Weak | Useful as control, not as adjustment |
| Publish as "demand-adjusted" metric | ❌ No | Misleading — implies calibration |

**A pressing-only intercept probability could be added as a *covariate* in
regression models** (e.g., "controlling for pressing difficulty"), but should
never be presented as a true expected-value model or "demand-adjusted accuracy."

---

## 3. What Would Proper Demand-Adjustment Require?

### 3.1 The Conceptual Framework

A proper demand-adjusted pressing accuracy metric answers:

> Given the pressing opportunities a defender faced, how many *should* they
> have won, and did they win more or fewer than expected?

Formally:
```
demand_adjusted = actual_press_success_rate − expected_press_success_rate
```

Where `expected_press_success_rate` comes from a **calibrated model** that maps
contextual features to the probability of a press resulting in a successful
outcome (interception, tackle, forced error).

### 3.2 Feature Engineering

A proper model would need per-pressing-event features:

| Feature | Source | Current pipeline? |
|---------|--------|-------------------|
| Distance to attacker at press initiation | Tracking (xy) | ✅ Available |
| Defender speed (magnitude) | Tracking | ✅ Available (v_mag) |
| Defender speed (direction relative to attacker) | Tracking (angle) | ✅ Available (angle) |
| Attacker speed | Tracking | ❌ Not computed per pair |
| Attacker direction | Tracking | ❌ Not computed per pair |
| Ball proximity | Tracking | ⚠️ Ball included but not feature |
| Defensive support (teammates nearby) | Tracking | ❌ Not computed |
| Body orientation | Tracking | ❌ Not in StatsPerform data |
| Pitch zone | Tracking (xy → zone) | ⚠️ Possible but not implemented |
| Game state (score, time) | Metadata | ❌ Not in pipeline |
| Attacker identity/skill proxy | Metadata | ❌ Not used |
| Defender identity | Tracking | ✅ Available |

**Missing key dimensions** even with tracking data alone:
- **Ball trajectory** — knowing where the ball is going matters more than
  defender–attacker kinematics
- **Teammate pressure** — a defender pressing alone vs. with support
- **Attacker context** — is the attacker facing goal, shielding, about to pass?

### 3.3 Outcome Labeling

The hardest gap: **we need to know whether each press succeeded or failed**.

Options:
1. **Event data (StatsPerform/StatsBomb):** Interception events, tackle events,
   or duel outcomes can serve as ground truth for press outcomes. Requires
   synchronised event data.
2. **Manual labelling:** Impractical for 30 matches.
3. **Heuristic from tracking:** Could infer "press failed" if attacker maintains
   possession >2s after press starts. Complex to implement and validate.

**None of these are currently available in the pipeline.**

### 3.4 Model Architecture

A reasonable model would be:

```
P(successful_press | X) = logistic_regression(X)  (interpretable baseline)
                        or gradient_boosted_tree(X)  (higher accuracy)
                        or neural_network(X)  (if large dataset)
```

Training data: all pressing events across all 30 matches, with features X and
outcome label y (success=1, failure=0).

Evaluation: held-out matches, calibration curves (reliability diagrams),
Brier score, AUC-ROC.

### 3.5 Literature Comparison

| Paper | Approach | Calibrated? | Data needed |
|-------|----------|-------------|-------------|
| Bekkers (2025) — Pressing Intensity | TTI physics model | No | Tracking only |
| Groom et al. (2026) — Defensive Role | ML expected positioning | Yes (ML) | Tracking |
| **This project (current)** | TTI + threshold | **No** | Tracking only |
| **What proper version needs** | Logistic/GBM on press features | **Yes** | Tracking + events |

The gap between what we have and what the literature supports:
- Bekkers' pressing intensity metric doesn't claim to measure accuracy — it
  measures *intensity* (frequency × speed × proximity)
- Groom et al. use a learned CDHMM for expected positioning, not pressing
- No published paper we've reviewed does "demand-adjusted pressing accuracy"
  in the way Jervis described — suggesting it's novel but requires proper
  empirical calibration

---

## 4. Pros/Cons: Physics-Based Intercept Probability vs. Learned Model

| Aspect | Physics-based (current) | Learned model (would need) |
|--------|------------------------|----------------------------|
| **Implementation effort** | ✅ Already implemented | ❌ 1–2 weeks (feature engineering + training + validation) |
| **Interpretability** | ✅ Every parameter is meaningful | ⚠️ Moderate (less transparent but can use SHAP) |
| **Calibration** | ❌ Not calibrated — sigmoid with arbitrary k | ✅ Proper probability calibration |
| **Transferability** | ✅ No data dependencies — works on any match | ⚠️ Requires same feature schema, retrain per league |
| **Sample size needed** | ✅ Works on a single frame | ❌ Needs ~1000+ pressing events per player group for stable estimates |
| **Outcome data needed** | ✅ None | ❌ Requires event data (interceptions, tackles, duel outcomes) |
| **Computation** | ✅ Lightweight (~3s per match) | ⚠️ Heavier (training + inference pipeline) |
| **Face validity** | ⚠️ Reasonable kinematic heuristic | ✅ Actually reflects empirical success rates |
| **Publishability** | ❌ Hard to defend as "accuracy" | ✅ Methodologically rigorous |

---

## 5. Recommendation

### Do Not Pursue "Demand-Adjusted Pressing Accuracy"

1. **The name is misleading.** You cannot "adjust" for pressing demand using an
   uncalibrated physics heuristic. Presenting it as demand-adjusted would
   (rightfully) draw Conor's skepticism.

2. **The gap to a proper version is large.** Without event data for outcome
   labels, the model cannot be calibrated. This is a fundamental data
   limitation, not a coding issue.

3. **Even the simple proxy is weak.** The pressing-only intercept probability
   distribution is highly skewed (median 0.015, mean 0.192), making the mean
   a poor summary statistic for "difficulty."

### Instead, Consider One of These

**Option A: Add pressing difficulty as a model covariate (recommended)**

Add `mean_intercept_prob_pressing` to the unified dataset. Include it as a
control variable in regression models testing the fatigue hypothesis:

```
pressing_accuracy ~ pressure_category + mean_intercept_prob_pressing + (1|player_id)
```

This controls for between-player differences in pressing difficulty without
misrepresenting the covariate as a calibrated adjustment.

**Option B: Reframe the pressing accuracy signal**

The current signal is already valid as-is: it measures **the fraction of
pressing actions where the defender's intercept probability exceeded 0.18.**
This is a real behavioural measure. It's just not "demand-adjusted." The
existing framing (comparing high-pressure vs. low-pressure blocks) is the
right approach — it's the fatigue-contrast design that does the adjustment,
not the metric itself.

**Option C: Future work — build the proper model**

If event data becomes available (interception events, tackle outcomes, or
duel outcomes synchronised with tracking), building a calibrated expected
pressing model would be:
- A publishable methodological contribution
- Feasible in ~1–2 weeks of work
- A genuine improvement over the literature

This should be scoped as a separate project, not a patch to the current
pipeline.

---

## Appendix A: Empirical Comparison (100k-frame sample)

Run against 100,000 frames of match 2215790 (all defenders, block 1):

### Current Aggregation (ALL frames, not just pressing)

| Player | N_presses | Pressing accuracy | `mean_intercept_prob` (ALL) | Notes |
|--------|-----------|-------------------|-----------------------------|-------|
| 40784 | 184 | 0.679 | 0.074 | Centre-back? |
| 78275 | 826 | 0.383 | 0.142 | High pressing volume |
| 97041 | 723 | 0.490 | 0.099 | |
| 102636 | 581 | 0.365 | 0.074 | |
| 107641 | 831 | 0.141 | 0.037 | Low accuracy despite volume |
| 116627 | 1042 | 0.205 | 0.049 | Highest pressing volume |
| 154048 | 653 | 0.609 | 0.149 | |
| 186796 | 0 | 0.000 | 0.000 | Goalkeeper? |
| 197017 | 387 | 0.065 | 0.024 | Near-zero accuracy |
| 477711 | 1092 | 0.300 | 0.106 | High volume, moderate accuracy |
| 477717 | 469 | 0.168 | 0.042 | |

### Comparison: ALL-frame vs Pressing-only intercept probability

| Metric | ALL frames | Pressing frames only |
|--------|-----------|---------------------|
| Mean intercept_prob | 0.074 | **0.192** |
| Median intercept_prob | 0.000 | **0.015** |
| Frames | 45,957 | 6,789 |

The pressing-only mean (0.192) is 2.6× higher than the all-frame mean (0.074),
but still pulled by a long tail — 50% of pressing frames have intercept_prob
< 0.015, meaning even the pressing-only mean is a poor summary of a heavily
skewed distribution.

---

## Appendix B: Code Snippet — How to Compute Pressing-Only Intercept Probability

If this covariate is added, the change to `aggregate_pressing_by_block()` in
`src/signals/pressing.py` is minimal:

```python
# In the grouped.agg() call, add:
mean_intercept_prob_pressing=("intercept_probability", lambda x: x[df.loc[df["is_pressing"]].index].mean())

# But this would need access to the full df within the agg context.
# A cleaner approach — compute before grouping:
subset["intercept_prob_pressing"] = np.where(
    subset["is_pressing"],
    subset["intercept_probability"],
    np.nan
)
# Then groupby and agg:
mean_intercept_prob_pressing=("intercept_prob_pressing", "mean"),
```

This would add a column `mean_intercept_prob_pressing` to the output alongside
the existing `mean_intercept_prob`. Both should be clearly documented:
- `mean_intercept_prob` = average over ALL frames (current, misleading)
- `mean_intercept_prob_pressing` = average over pressing frames only (useful as covariate)

---

*This document is a honest audit of the data pipeline's capabilities.
Recommendation: do not claim "demand-adjusted" without proper calibration.
Add pressing-only intercept as a covariate at most.*
