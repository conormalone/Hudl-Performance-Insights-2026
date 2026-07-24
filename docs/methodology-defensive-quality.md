# Defensive Quality Methodology — Draft Framework

**Project:** Focus Fatigue — Cognitive Fatigue Detection in Football Defence
**Status:** Phase 0 draft (21 Jun 2026)
**Key references:** Groom et al. (2601.00748), Bekkers (2501.04712)

---

## Overview

Two-model architecture. Model 1 quantifies exposure (when is a defender under cognitive load). Model 2 measures the behavioural response (do their defensive quality signals degrade during those periods).

```
         Model 1                          Model 2
  ┌──────────────────┐           ┌────────────────────────┐
  │ Pressure Exposure │ ──────►  │ Defensive Quality      │
  │ (load indicators) │  context │ (5 behavioural signals) │
  └──────────────────┘           └────────────────────────┘
                                         │
                                         ▼
                              Fatigue score per 5-min block
```

---

## Model 1 — Pressure Exposure

Computes weighted time-under-pressure per 5-minute block for each defender.

### Load indicators (per frame)

| Indicator | Definition | Data source |
|-----------|-----------|-------------|
| **Opponent proximity** | Number of attackers within radius (distance calibrated per team, ~5–10m) | Tracking |
| **Defensive depth** | Distance from own goal line (absolute) | Tracking |
| **Reorientation frequency** | Number of sharp heading changes (>45° in <1s) in trailing window | Tracking (velocity vector) |
| **Transition count** | Number of possession changes in defender's zone in trailing window | Event data |

### Baseline construction

- Each player builds their own per-match baseline from their first 15 minutes (minimal fatigue assumption).
- Player-level: running average of each indicator across all available matches.
- Rotation players (<180 min across dataset): fall back to position-level averages.

### Time-under-pressure calculation

Per 5-minute block:

```
weighted_pressure = Σ(frame_weight × active_indicators)
                   ─────────────────────────────────────
                         frames_in_block

where frame_weight = 1 + sum(indicator_i / baseline_i)
      active_indicators = indicators exceeding personal threshold
```

Blocks are then ranked: top-quartile blocks = "high pressure", bottom-quartile = "low pressure" (control).

---

## Model 2 — Five Defensive Quality Signals

Each signal is computed per 5-minute block and compared between high-pressure and low-pressure blocks. A defender showing degradation in high-pressure blocks *without* corresponding degradation in raw running metrics is flagged for cognitive fatigue.

### 1. Positional Drift

**What it measures:** Deviation from expected position given team shape and ball location.

**Methodology:** For each frame, compute:

```
expected_position = f(team centroid, ball location, formation template)
drift = ||actual_position - expected_position||
```

- Formation template learned per match via EFPI-style approach (Bekkers, 2506.23843) — cluster player positions during settled possession to infer shape.
- Drift is normalised by the defender's own baseline (first 15 min of match).
- High drift = defender losing structural discipline.

**Reference:** Groom et al. infer expected positions from CDHMM zonal emissions. Our equivalent uses formation-aware expected positions from tracking data.

**Confound control:** Drift during opposition possession is expected. We condition on game state — measuring drift *only* during structured defensive phases (when team is set, not during transitions).

---

### 2. Shift Latency

**What it measures:** Time between a significant ball movement or opponent run and the defender's corrective movement.

**Methodology:**

```
trigger = ball velocity spike OR opponent run > threshold
shift_latency = time(trigger) - time(defender acceleration toward new target)
```

- Detect trigger events from tracking data (ball speed changes, opponent runs beyond a threshold speed toward dangerous space).
- Measure defender response: latency from trigger to the frame where their velocity vector changes by >30° toward the new threat.
- Aggregated per 5-min block as mean and 90th percentile latency.

**Reference:** Bekkers' pressing model uses reaction time (τᵣ) as a fixed parameter. We estimate it dynamically per defender per block — higher latency = slower cognitive processing.

**Expected fatigue signal:** Shift latency increases as cognitive fatigue accumulates, even when sprint speed is maintained.

**Confound control:** Compare to the same defender in low-pressure blocks of the same match, controlling for match minute.

---

### 3. Pressing Decision Accuracy

**What it measures:** Was the defender's pressing action appropriate for the situation?

**Methodology:**

```
pressing_event = defender accelerates toward attacker within 2s of opponent receival
accuracy_score = did pressing_event occur when intercept_prob > threshold?
```

- Compute pressing probability per frame using Bekkers' time-to-intercept formula:

  ```
  T_defender = τᵣ + distance_to_intercept / v_max + direction_penalty
  p_intercept = 1 / (1 + exp(-π/(√3·σ) · (T_limit - T_defender)))
  ```

- A pressing decision is "correct" when the defender presses a ball carrier or imminent receiver with p_intercept > 0.3 (configurable threshold).
- A pressing decision is "incorrect" (wasted energy / dragged out of shape) when they press with p_intercept < threshold.
- Accuracy = (correct presses) / (total pressing actions) per block.

**Expected fatigue signal:** Decision accuracy degrades while physical capacity (sprint speed) holds steady. Defender presses when they shouldn't, or fails to press when they should.

**Reference:** Beckers' Active Pressing threshold (speed < 2m/s → no pressing) is directly used here to filter noise.

---

### 4. Spatial Awareness

**What it measures:** Coverage of dangerous space relative to ball and teammates.

**Methodology:**

```
danger_value = pitch_control_value(opponent) at defender's location
coverage_score = 1 - danger_value at defender's actual position
```

- Use a simplified pitch control model (Spearman) to compute the opponent's control probability at each point on the pitch.
- For each defender, compute the danger value at their location. A well-positioned defender covers dangerous space (low danger value). A poorly positioned defender leaves gaps.
- Compare to: "what would the danger value be if the defender was in their expected position?" (from Signal 1's expected position).

**Reference:** Groom et al. use ghosting — compare actual defender to a role-conditioned counterfactual. Our equivalent: compare actual spatial coverage to what the defender's expected-position ghost would provide.

**Expected fatigue signal:** Gap between actual coverage and ghost coverage widens over the match, especially in high-pressure blocks.

---

### 5. Transition Recognition Time

**What it measures:** How long after a turnover does the defender recognise and respond to the transition.

**This is the most sensitive signal** — transition recognition is purely cognitive (processing the change) while the physical recovery run that follows is a separate measure.

**Methodology:**

```
turnover_frame = event_data identifies change of possession
recognition_frame = first frame where defender velocity vector points toward own goal
                   AND defender is accelerating (speed > 0.5 m/s and increasing)
transition_latency = recognition_frame - turnover_frame
```

- Separately measure **recovery sprint speed** (peak velocity during the recovery run). The gap between transition latency and recovery speed is the cognitive-physical disconnect.
- Aggregated as mean and max latency per 5-min block.

**Expected fatigue signal:** Transition recognition time increases over the match while peak recovery sprint speed does not change. This is the strongest single indicator of cognitive (not physical) fatigue.

**Confound control:** Control for surprise transitions vs. expected ones (e.g., a long ball that was obviously coming vs. a dispossession in midfield). Use event context: transition type flag from event data.

---

## Signal Aggregation

### Option A — Keep signals separate (recommended initially)

Each signal is its own dependent variable. Report which signals degrade and which don't. This is more informative for coaches ("his transition recognition is slipping but his pressing is fine").

### Option B — Composite Fatigue Index

```
fatigue_index = (z_drift + z_latency + z_pressing + z_awareness + z_transition) / σ_total
```

Where each z-score is the defender's value in a high-pressure block relative to their low-pressure baseline. Weighted by signal reliability (transition recognition gets highest weight).

---

## Validation Strategy (from Q4)

Without experimental fatigue induction, we frame findings as:

1. **Convergent evidence:** Do multiple signals point in the same direction per block?
2. **Discriminant validity:** Do the signals *not* correlate with raw running metrics (distance covered, sprint count)?
3. **Temporal pattern:** Does fatigue increase monotonically across match halves within high-pressure blocks?
4. **Substitution validation:** For players substituted off, do our signals spike in the 15 minutes before substitution?

---

## Implementation Plan

| Step | What | Dependencies |
|------|------|-------------|
| 1 | Load and canonicalise tracking data (pitch alignment, synchronise with events) | Data from Hudl |
| 2 | Compute formation template per match (EFPI-style clustering) | Step 1 |
| 3 | Implement Model 1 — Pressure Exposure | Step 1 |
| 4 | Implement Signal 1 — Positional Drift | Steps 1, 2 |
| 5 | Implement Signal 5 — Transition Recognition (easiest, highest-value) | Step 1 + event data |
| 6 | Implement Signal 3 — Pressing Accuracy (Bekkers method) | Step 1 |
| 7 | Implement Signal 2 — Shift Latency | Step 1 |
| 8 | Implement Signal 4 — Spatial Awareness (simplified pitch control) | Steps 1, 2 |
| 9 | Run both models, aggregate per block, test validation hypotheses | Steps 3–8 |
| 10 | Sensitivity analysis: vary thresholds, compare results | Step 9 |

**Recommended order:** Signal 5 (Transition Recognition) first — clearest cognitive-physical distinction, lowest implementation complexity. Then Signal 3 (Pressing Accuracy — Bekkers code may be reusable). Then Signals 1, 2, 4.

---

## Open Questions

1. What is the optimal time window? 5 minutes per block, or adapt to natural phases of play?
2. How do we handle substitutions? New player entering resets the fatigue clock.
3. What about centre-backs vs. full-backs — different fatigue profiles? Position-level stratification?
4. Should we normalise by opposition pressing intensity (how much pressure the opponent applies)?
5. For transition recognition — do we need optical tracking or is broadcast tracking sufficient to detect velocity changes?

---

*Draft: 21 June 2026 | Jervis 🧠*
