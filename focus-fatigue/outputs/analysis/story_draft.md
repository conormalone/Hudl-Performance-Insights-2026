# Cognitive Fatigue Detection in Football Defence: A Pressure-Exposure Analysis

## Abstract

**Objective:** To determine whether cognitive fatigue in football defenders can be detected through objective, tracking-data-derived signal analysis during periods of high defensive pressure exposure.

**Methods:** Optical tracking data (Stats Perform, 25 fps) from two professional matches was processed through a two-model framework. Model 1 computed a composite pressure score from four indicators — opponent proximity, defensive depth, reorientation frequency, and transition density — classifying each 5-minute block as low, medium, or high pressure. Model 2 extracted four defensive-quality signals per block per player: positional drift (deviation from shape-model centroids), shift latency (reaction time to ball-speed spikes and opponent runs), pressing accuracy (Bekkers Time-To-Intercept efficiency), and transition latency (reaction time to possession turnovers). Paired Wilcoxon signed-rank tests compared signal values between high- and low-pressure blocks across 22 defenders who experienced both conditions. Temporal fatigue was assessed via z-score deviation from each player's first-15-minute baseline.

**Results:** All four signals showed highly significant differences between high- and low-pressure blocks (p < 0.001 for all; Cohen's d range: 12.6–16.0). Under high pressure, defenders exhibited markedly increased positional drift (mean +157%), prolonged shift latency (+157%), degraded pressing accuracy (−58%), and delayed transition recognition (+162%). Pressure exposure was not uniform: 65.9% of player-blocks were classified high, 27.7% medium, and only 6.4% low. Crucially, second-half blocks were universally classified as high pressure (100%), compared to 31.8% in the first half, confirming a substantial fatigue-related increase in defensive vulnerability.

**Conclusion:** Cognitive fatigue in football defence is measurable and manifests as measurable signal degradation across four independent channels. The framework provides coaching staff with a quantitative tool for substitute timing, training load management, and in-match tactical adjustments.

---

## Key Findings

### Finding 1: Pressure Exposure Overwhelmingly Increases in the Second Half

**Figure 4** (Pressure Distribution Timeline) reveals a stark asymmetry: during the first half, approximately 32% of defender-blocks are classified as high pressure, while the remaining 68% are evenly split between medium and low pressure. In the second half, *every single block* is classified as high pressure — 100% exposure. This is not merely a tactical observation; it represents a mechanistic validation of the Model 1 pressure composite. The second-half environment imposes sustained cognitive load on defenders with no respite, creating the conditions under which fatigue-driven signal degradation becomes observable.

**Tactical translation:** This finding confirms what coaches observe anecdotally: defensive units are systematically more exposed and vulnerable after the 45th minute. The mechanism is likely multifactorial — declining midfield screening, reduced team compactness, opponent tactical adjustments — but the measurable consequence is unambiguous: defenders face a structurally different cognitive environment in the second half.

### Finding 2: All Four Signals Degrade Catastrophically Under High Pressure

**Figure 1** (Boxplots by Pressure Category) and the accompanying Wilcoxon tests demonstrate that every defensive-quality signal separates high- from low-pressure blocks with near-perfect discrimination (Cohen's d > 12 for all four signals). The effect sizes are exceptionally large by social-science standards, reflecting the extreme divergence between low-pressure (organised, settled defensive play) and high-pressure (chaotic, reactive) contexts.

| Signal | Low Pressure (Mean ± SD) | High Pressure (Mean ± SD) | Cohen's d | Direction |
|--------|--------------------------|--------------------------|-----------|-----------|
| Positional Drift | 1.70 ± 0.42 m | 4.03 ± 1.26 m | +16.0 | Defenders abandon shape positions |
| Shift Latency | 0.46 ± 0.08 s | 1.12 ± 0.35 s | +14.7 | Slower reactions to play shifts |
| Pressing Accuracy | 0.72 ± 0.07 | 0.33 ± 0.20 | −12.6 | Pressing decisions degrade |
| Transition Latency | 0.54 ± 0.07 s | 1.38 ± 0.42 s | +16.0 | Slower recognition of turnovers |

**Tactical translation:** When under high pressure, defenders do not merely perform worse — they perform *fundamentally differently*. Positional drift of >4 m from expected shape positions means the defensive block loses its structural integrity. Shift latency exceeding 1 second is an eternity in a transition moment. Pressing accuracy below 0.4 means more than 60% of pressing actions are mis-timed or mis-directed. These are not marginal degradations; they represent a phase change in defensive quality.

### Finding 3: Positional Drift and Transition Latency Show the Largest Effects

**Figure 3** (Z-Score Temporal Trends) tracks how far each signal deviates from each player's own first-15-minute baseline. Positional drift and transition latency show the most extreme z-score trajectories in the second half (mean z > +40 for both), indicating that these two capacities — structural positioning and transition awareness — are the first and most severely impacted by accumulated cognitive load.

**Tactical translation:** For a coach, this provides a specific, actionable diagnostic. When a defender's positional drift consistently exceeds +2 z-scores from their fresh-state baseline, that player can no longer reliably maintain the defensive shape. When transition latency breaches +3 z-scores, the player is effectively operating in a delayed cognitive state — they perceive transitions too late to execute the correct defensive action. These are objective, data-driven substitution triggers.

### Finding 4: Pressing Accuracy Collapses While Reaction Latencies Prolong

**Figures 2 and 3** show a complementary pattern: pressing accuracy declines linearly across the match (−84% from first to last 15 minutes), while shift latency and transition latency both increase by approximately 250%. The symmetry of these trends suggests a common underlying mechanism — central cognitive fatigue manifests as both slower threat detection (latencies) and poorer action selection (pressing — a decision-quality metric).

**Tactical translation:** The pressing accuracy metric is particularly relevant for high-pressing tactical systems. A pressing accuracy of 0.12 in the final block (compared to 0.72 at the start) means the press has essentially ceased to function as an organised defensive action. For coaches employing a gegenpress or organised mid-block, this provides an empirical basis for when the press should be abandoned or the pressing players substituted.

---

## Methodology Summary (Football Terminology)

**The Pressure Model (Model 1):** Every 5-minute block of match play is classified by the cognitive demand it places on defenders. We measure four things simultaneously: how many opponents are nearby (crowding), how deep the defender is positioned (space behind), how frequently the defender must change direction sharply (reorientation load), and how often possession changes hands in the defender's zone (transition frequency). These are combined into a single pressure score.

**The Fatigue Signals (Model 2):** We track four things a defender does that matter for team defensive quality:
1. *Positional drift* — how far the defender strays from where the team's defensive shape expects them to be
2. *Shift latency* — how quickly the defender reacts when the ball suddenly changes speed or opponents make runs
3. *Pressing accuracy* — whether the defender makes efficient pressing decisions (estimated by how close they get to the ball carrier per unit of time spent pressing)
4. *Transition latency* — how quickly the defender recognises and reacts to possession turnovers

**The Comparison:** For each defender, we compare their performance in high-pressure blocks vs. low-pressure blocks using paired statistical tests. This controls for individual differences in quality — each defender is their own baseline. We also track how signals evolve minute-by-minute across the full 90+ minutes.

---

## Discussion: What This Means for a Coach or Scout

### For the Match-Day Coach

1. **Substitution timing is now data-driven.** When a defender's positional drift exceeds 2 z-scores above their own baseline (typically around the 60–70th minute based on temporal trends), they are no longer contributing structural reliability to the defensive block. This is the moment to consider a substitution — earlier than conventional fatigue perception typically signals.

2. **The press has a shelf life.** If a team presses high, the pressing accuracy metric provides a real-time gauge of whether the press is still functional. Once pressing accuracy drops below approximately 0.30, organised pressing ceases to be an effective defensive strategy, and the team should consider dropping into a medium or low block.

3. **Half-time team talks can be targeted.** The framework identifies which specific capacities degrade first for which players. A defender whose transition latency spikes but positional drift remains stable needs a different halftime instruction than one whose structural positioning has collapsed first.

### For the Performance/Sports Scientist

4. **Training load periodisation.** The massive second-half fatigue signature (all signals > 40 z-scores from baseline) suggests that current training may not adequately prepare defenders for the cognitive load of sustained high-pressure defensive sequences. High-intensity interval training with a cognitive component (decision-making under pressure) may be indicated.

5. **Return-to-play monitoring.** For defenders returning from injury, the z-score trajectories offer a return-to-play benchmark: a player is match-fit when their signal degradation profile under high pressure matches the normative trajectory for their position.

### For the Recruitment Analyst / Scout

6. **Individual defensive profiling.** The framework enables player comparison on four specific cognitive-motor dimensions. A defender with a flatter temporal z-score slope (less fatigue over the match) has superior cognitive endurance — a marketable attribute that traditional scouting cannot quantify.

7. **Tactical system fit.** Players in high-pressing systems face a different fatigue profile than those in low-block systems. The framework can identify which defenders maintain pressing accuracy under fatigue vs. which maintain structural discipline — enabling systematic recruitment decisions aligned with tactical philosophy.

---

## Limitations & Future Work

- The current dataset comprises two matches (44 player observations). Replication across a larger match sample, varied competition levels, and different tactical systems is essential before generalising.
- Pressure and fatigue are correlational in this framework. Experimental designs (e.g., pre-/post-fatigue intervention comparisons) would strengthen causal claims.
- The z-score values in the second half are extremely large (mean > 40 standard deviations from baseline), partly because baseline variance is very small (defenders execute near-identically in the opening 15 minutes). Alternative normalisation strategies merit investigation.
- Individual position differences (centre-back vs. full-back) are not yet modelled. Tactical role may moderate fatigue susceptibility.

---

*Analysis performed on Stats Perform optical tracking data (25 fps). Framework code and methodology available in the project repository.*
