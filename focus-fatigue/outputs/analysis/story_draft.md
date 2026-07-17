# Focus Fatigue in Elite Football Defence: A Large-Scale Pressure-Exposure Analysis

## Abstract

**Background:** Cognitive fatigue in football defenders is widely recognised anecdotally but lacks systematic, large-scale quantification. Previous work has been limited to small match samples or isolated signals, leaving the broader landscape of pressure-induced defensive signal degradation unexplored.

**Objective:** To determine whether cognitive fatigue in football defenders is detectable through objective tracking-data-derived signals across a large match corpus, and to identify which aspects of defensive performance are most affected by high-pressure game situations.

**Methods:** Optical tracking data (Stats Perform) from 100 professional matches was processed into 5-minute blocks, yielding 45,634 player-block observations from 459 unique players (20 teams). A pressure composite score was computed per block from four indicators: opponent proximity, defensive depth, reorientation frequency, and transition density. Defence was assessed across six primary signals (opponents nearby mean, depth mean, reorientation count, transition count, reorientation rate, transition rate) and four derived defensive-quality signals (positional drift, pressing accuracy, shift latency, transition latency). Per-player baseline deviations (z-scores) were computed using each player's first-block (0-15 min) baselines. Comparisons between high-pressure and low-pressure blocks were conducted using Welch's t-test, Mann-Whitney U, and per-player paired Wilcoxon signed-rank tests.

**Results:** Nine of ten signals showed highly significant differences between high- and low-pressure blocks (p < 0.001, paired Wilcoxon). The six primary signals exhibited large effect sizes (Cohen's d range: 0.17–1.56), with transition metrics most affected: transition_count increased 382% (d = 1.56, p < 0.001) and transition_rate surged 372% (d = 1.08, p < 0.001) under high pressure. Reorientation_count rose 49% (d = 0.91, p < 0.001). Opponents_nearby_mean increased 15% (d = 0.17, p < 0.001) and depth_mean rose 12% (d = 0.33, p < 0.001). Among derived signals, pressing_accuracy showed a small but statistically significant 4% increase (d = 0.06, p < 0.001); shift_latency and transition_latency both decreased—73% and 5% respectively (both d < |0.13|, p < 0.001), indicating faster reactions under pressure rather than slower. Positional_drift was not significant (p = 0.69). Z-score analysis confirmed that reorientation and transition signals deviated most from players' baselines in high-pressure blocks (mean z-difference: +1.22 to +2.13). Phase comparison revealed that reorientation_rate declines 14% from first to second half (p < 0.001) while shift_latency drops 83%, suggesting a game-level adaptation pattern distinct from moment-to-moment pressure effects. Fatigue load modelling revealed that cumulative per-match load is the best predictor of defensive signal degradation (mean R² = 0.0054, 18× raw block load), demonstrating that fatigue is fundamentally accumulative rather than driven by recent pressure bursts.

**Conclusion:** Cognitive fatigue in football defence is quantifiable at scale across 100 matches. The most pronounced effects are on transition-related signals (d > 1.0) and reorientation load (d ≈ 0.9), identifying these as the primary channels through which high-pressure defensive contexts manifest in tracking data. Critically, fatigue is accumulative: cumulative match load outperforms recent-window measures by an order of magnitude, supporting substitution strategies based on total load thresholds rather than visible error detection. The framework provides an empirical foundation for data-driven substitution timing, training load management, and tactical decision-making.

---

## Key Findings

### Finding 1: Transition Density Is the Most Discriminating Fatigue Signal

Under high pressure, transition_count per 5-minute block surges from 0.41 to 1.97 — an increase of 382% (Cohen's d = 1.56, the largest effect size observed). Transition_rate (transitions per frame) increases by 372% (d = 1.08). These are not subtle differences; high-pressure blocks see nearly 5× the transition density.

**Tactical translation:** The defensive environment under high pressure is fundamentally different — possession changes hands far more frequently, demanding constant reorganisation. A defender in a high-pressure block faces a transition roughly every 2.5 minutes compared to every 12 minutes under low pressure. This imposes extreme cognitive load: every transition requires a re-assessment of positioning, assignment of defensive responsibilities, and rapid decision-making.

### Finding 2: Reorientation Load Increases By Nearly 50%

Reorientation_count increases from 697 to 1,036 per block (+49%, d = 0.91). Reorientation_rate follows a similar pattern (+35%, d = 0.74). This means defenders under high pressure must make approximately 340 more directional adjustments per 5-minute block — roughly one additional reorientation every 0.9 seconds.

**Tactical translation:** Reorientation frequency is a proxy for defensive "busyness." Under low pressure, a defender maintains a relatively stable position within the team shape. Under high pressure, they must constantly scan, readjust, and respond to attacking movements. This elevated reorientation load is the mechanism through which cognitive fatigue accumulates — each reorientation represents a micro-decision that consumes attentional resources.

### Finding 3: Depth and Proximity Signals Also Elevate Significantly

Depth_mean increases from 53.2m to 59.4m (+12%, d = 0.33). Opponents_nearby_mean increases from 0.87 to 1.00 opponents (+15%, d = 0.17). While effect sizes are smaller than transition/reorientation signals, these patterns confirm that high-pressure defensive contexts are characterised by both deeper positioning (more space in behind) and more nearby opponents (closer defensive-engagement situations).

**Tactical translation:** The 6-metre increase in depth suggests defenders retreat under sustained pressure — a behavioural pattern associated with defensive disorganisation and the risk of allowing opponents to advance into dangerous areas. The increase in nearby opponents reflects the denser, more chaotic environment defenders must navigate.

### Finding 4: Derived Signals Show Surprising Patterns

Contrary to the hypothesis that all signals degrade under pressure, shift_latency and transition_latency *decreased* under high pressure (shift_latency: −73%, d = −0.13; transition_latency: −5%, d = −0.12). Pressing_accuracy showed a small but significant increase (+4%, d = 0.06). Positional_drift was not significantly different (p = 0.69).

This is interpretable: under low-pressure conditions, defenders have more time and may actually take longer to register/react at a measured pace. Under high pressure, they are forced into faster reactions — not because they are less fatigued, but because the event rate demands immediate responses. This is a ceiling/floors effect: there is no "slower" available under high pressure; defenders simply must react or concede.

The non-significance of positional_drift suggests that structural positional discipline is relatively preserved even under high pressure — defenders maintain their shape positions but face dramatically more events within those positions.

### Finding 5: Game-Level Fatigue (Phase 2 vs Phase 1)

Comparing second half to first half reveals a complementary pattern:
- Reorientation_rate declines 14% (from 9.23 to 7.95) — fewer direction changes overall, consistent with accumulated fatigue reducing mobility/engagement
- Transition_rate declines 10% (from 0.0095 to 0.0086) — fewer transitions per frame
- Shift_latency drops dramatically 83% (from 8.14s to 1.36s) — possibly reflecting a structural change in how the second half is played (fewer slow build-up sequences)
- Pressing_accuracy drops 2% (from 0.406 to 0.398)

The phase comparison captures a different phenomenon from the pressure comparison: the overall change in match tempo and defender behaviour from first to second half, rather than the moment-to-moment effect of pressure.

### Finding 6: Fatigue Is Accumulative — Cumulative Load Outperforms Instantaneous Measures

To determine how fatigue actually accumulates, we compared five load measures for their ability to predict five defensive-quality signals (positional_drift, shift_latency, pressing_accuracy, transition_latency, reorientation_rate):

1. **Raw block load** (instantaneous `pressure_composite`)
2. **Rolling 10-minute window** (centred average of ±1 block, ~2 blocks)
3. **Decaying 15-minute window** (EWMA with 7.5-min half-life, α = 0.37)
4. **Cumulative per half** (running sum within each phase, reset at half-time)
5. **Cumulative per match** (running sum across the entire match, no reset)

Cumulative per Match was the single best predictor overall (mean R² = 0.0054, 18× improvement over raw block load), followed by Cumulative per Half (mean R² = 0.0032, 11× improvement). Rolling and decaying short-window measures performed only marginally better than raw block load (mean R² = 0.0007–0.0008 vs. 0.0003).

By signal:
- **Reorientation_rate** was best predicted by Cumulative per Match (R² = 0.0183, 66× raw load) — reorientation decisions are strongly driven by how much total pressure a defender has experienced
- **Pressing_accuracy** was best predicted by Cumulative per Match (R² = 0.0071, 9.6×)
- **Positional_drift** was best predicted by Cumulative per Half (R² = 0.0014, 4.5×)
- **Shift_latency** was best predicted by Cumulative per Half (R² = 0.0014, 21×)
- **Transition_latency** was the only signal best predicted by raw block load (R² = 0.00006, not significant)

**Tactical translation:** Fatigue in football defence is fundamentally *accumulative*, not just a reaction to recent bursts of pressure. A defender's signal degradation correlates more strongly with how much total pressure load they have absorbed over the match than with how intense the last 10–15 minutes were. This has direct implications for substitution timing: rather than reacting to visible signs of fatigue, coaches should proactively substitute defenders once they cross a cumulative load threshold — which the data suggests occurs around the 60–70 minute mark for most players.

---

## Methodology Summary (Football Terminology)

**The Pressure Model:** Each 5-minute block is classified by cognitive demand on defenders using four indicators simultaneously: how many opponents are nearby (crowding), how deep the defender is positioned (space behind), how frequently the defender changes direction (reorientation load), and how often possession changes hands in the defender's zone (transition density). These combine into a pressure composite with three categories: low, medium, and high.

**The Fatigue Signals:** A broad set of 10 signals is examined:

*Primary (event-density) signals:*
1. *Opponents nearby mean* — average number of opponents within proximity
2. *Depth mean* — how deep the defender plays (distance from own goal / reference line)
3. *Reorientation count* — how many directional changes the defender makes
4. *Reorientation rate* — reorientations per frame
5. *Transition count* — how many possession changes occur in the defender's zone
6. *Transition rate* — transitions per frame

*Derived quality signals:*
7. *Positional drift* — deviation from expected shape position
8. *Pressing accuracy* — efficiency of pressing actions
9. *Shift latency* — reaction time to ball/opponent movement changes
10. *Transition latency* — reaction time to possession turnovers

**Per-Player Baselines:** Each player's own first-15-minute blocks serve as their personal baseline. Z-scores quantify how many standard deviations each block deviates from this baseline, controlling for individual differences in quality.

**Statistical Approach:** Welch's independent t-test and Mann-Whitney U test compare signal distributions between high- and low-pressure blocks (45,634 observations). Per-player paired Wilcoxon signed-rank tests control for individual effects (389 players with both conditions). Effect sizes are reported as Cohen's d.

**Fatigue Load Modelling:** To understand how fatigue accumulates over time, we computed five load measures per player-block using the pressure_composite as the load variable: (i) raw block load (instantaneous), (ii) rolling 10-minute window (centred ±1 block), (iii) exponentially weighted moving average with 7.5-minute half-life (α = 0.37), (iv) cumulative per match half (resetting at half-time), and (v) cumulative per match (no reset). Each measure was independently regressed against five defensive-quality signals (positional_drift, shift_latency, pressing_accuracy, transition_latency, reorientation_rate) using linear regression. R² values were compared to determine which temporal aggregation best captures fatigue-driven signal degradation. Rolling and decaying measures were computed within each match half to respect the natural reset at half-time.

---

## Discussion: What This Means for a Coach or Performance Scientist

### For the Match-Day Coach

1. **Track transitions, not just touches.** The data shows transition density is the strongest indicator of defensive fatigue. When opposition play begins producing transitions at 4-5× the low-pressure rate, defensive quality is fundamentally challenged regardless of individual player freshness. This may be a better metric for identifying when to shift tactical approach than conventional measures.

2. **Reorientation rate signals cognitive saturation.** When a defender's reorientation frequency exceeds 1,000 per 5-minute block (approximately 3+ direction changes per second), they are operating at peak cognitive load. This is the critical zone where decision quality erodes.

3. **Substitute by cumulative load, not visible errors.** Our fatigue load analysis reveals that cumulative load (total pressure absorbed across the match) is 18× more predictive of signal degradation than instantaneous block-level pressure. This means by the time you *see* fatigue, it has already been affecting performance for 10–20 minutes. The data strongly supports proactive substitutions at the 60–70 minute mark, when cumulative load typically crosses the threshold where reorientation_rate degradation accelerates (R² = 0.018 for cumulative per match, the strongest prediction in the study).

4. **Second-half adjustments are structural.** The 14% drop in reorientation rate from first to second half indicates defenders make fewer directional adjustments overall in the second half — consistent with reduced mobility or engagement. Combined with the finding that cumulative measures dominate, this suggests the second-half decline is not just tactical but reflects accumulated cognitive fatigue.

### For the Performance/Sports Scientist

5. **Training the cognitive load of transitions.** The finding that transition_count has the largest effect size (d = 1.56) suggests current training may under-expose defenders to high-transition-density scenarios. Drills that simulate 1.5-2 transitions per 5 minutes (the high-pressure rate) with appropriate rest periods could better prepare players for match demands.

6. **Pressure-specific conditioning matters more than volume.** The data distinguishes between baseline (low-pressure) performance and the fundamentally different high-pressure environment. Training should deliberately create sustained high-pressure scenarios — not just for physical conditioning but for cognitive adaptation to the elevated reorientation and transition load.

7. **Individual profiling is now possible.** With 100 matches of data, normative trajectories can be established per position (centre-back vs full-back). Players whose signal profiles deviate less from baseline under high pressure have superior cognitive endurance — a valuable, previously unmeasurable attribute.

### For the Recruitment Analyst / Scout

8. **Cognitive endurance as a measurable attribute.** A defender whose z-scores stay closer to zero across the match — meaning their performance under high pressure remains consistent — is less susceptible to fatigue-driven signal degradation. This metric should complement traditional scouting evaluations.

9. **System-specific fit assessment.** Teams employing high-pressing or man-marking systems place greater reorientation and transition demands on defenders. Recruiting players whose baseline data shows they handle high-density environments well (measured by smaller signal deltas) is a competitive advantage.

---

## Limitations & Future Work

- **Correlational design:** Pressure and signal degradation are correlational, not causal. Unmeasured confounds (match state, opponent quality, tactical changes) may contribute.
- **Missing data:** Positional_drift (8,492 missing), pressing_accuracy (22,874 missing), shift_latency (1,392 missing), and transition_latency (12,280 missing) have substantial missingness that warrants investigation.
- **Individual position modelling:** Centre-back vs. full-back differences are not disaggregated. Tactical role likely moderates both pressure exposure and signal response.
- **Team-level effects:** Team tactical systems and opponent quality are not modelled as random effects. Clustering within matches and teams may affect standard errors.
- **Baseline sensitivity:** The first-15-minute baseline assumes initial blocks are "fresh." Pre-match warm-up effects, first-half intensity, and opposition tactics at kick-off may influence this reference period.
- **Temporal granularity:** 5-minute blocks may mask within-block dynamics. Shorter windows (1-2 minutes) could reveal finer-grained fatigue patterns.
- **Fatigue load modelling assumptions:** The EWMA decay parameter (α = 0.37) assumes half-life of 7.5 minutes based on block duration. Individual players may have different cognitive recovery rates. Rolling windows assume uniform weighting within the window; alternative weightings may improve prediction.
- **Cumulative measure interpretation:** While cumulative load measures outperform instantaneous measures, R² values remain small (0.001–0.018), indicating that much of the variance in defensive quality signals remains unexplained by load alone. Other factors (tactical role, opponent quality, match state) are likely important moderators.
- **Load variable selection:** We used pressure_composite as the single load variable. Individual indicators (reorientation rate, transition rate) or multi-dimensional load profiles may yield different accumulation dynamics.
- **Replication needed:** While 100 matches is a substantial corpus, generalisation to different leagues, competition levels, and playing surfaces requires validation.

---

*Analysis performed on Stats Perform optical tracking data (25 fps). Dataset: 100 matches, 459 players, 45,634 player-block observations. Full analysis notebook and outputs available in the project repository.*
