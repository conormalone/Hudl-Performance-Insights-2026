# Literature Review — Cognitive Fatigue in Football Defence

**Project:** Focus Fatigue
**Status:** Finalised (24 Jun 2026)
**Papers reviewed:** 10 full-text PDFs + 10+ additional abstracts
**Next step:** Ready for citation in paper methodology section

---

## 1. Scope & Approach

This review covers the intersection of three research streams:
1. **Defensive analysis in football** — how off-ball defence is evaluated
2. **Spatio-temporal methods** — tracking data analysis, formation identification, pressing
3. **Fatigue & load monitoring** — cognitive vs. physical fatigue, match-induced decrements

Our project sits at the centre: using spatio-temporal tracking + event data to detect *cognitive* (not physical) fatigue in defenders via behavioural signals.

---

## 2. Defensive Analysis — The Underdeveloped Frontier

### 2.1 The Off-Ball Problem

Attacking metrics are mature — xG, VAEP, pass valuation — but defence is fundamentally harder because good defending often prevents events from happening (Tuyls et al., 2020; Merhej et al., 2021). Traditional counts (tackles, interceptions) miss the subtle ways defenders constrain attacking options.

**Groom et al. (2026)** — *A Machine Learning Framework for Off Ball Defensive Role and Performance Evaluation in Football* [arXiv:2601.00748]

This is the most directly relevant paper in our collection. Key contributions:
- **CDHMM for tactical roles:** A covariate-dependent Hidden Markov Model that infers man-marking vs. zonal assignments on corner kicks directly from tracking data — no manual labels needed.
- **Role-conditioned ghosting:** A counterfactual framework that replaces a defender with an "average" player in the same tactical role, enabling context-aware performance evaluation.
- **Key limitation for our work:** Focused exclusively on corner kicks (highly structured). Our project extends this logic to open-play defensive phases.

**Relevance to Focus Fatigue:** The ghosting methodology (Signal 4 — Spatial Awareness) directly inspired our approach. We compare actual defender positioning to a counterfactual "expected position" from team shape, analogous to Groom et al.'s role-conditioned ghosts.

### 2.2 Pressing & Defensive Action

**Bekkers (2025)** — *An Intuitive Measure for Pressing in Soccer* [arXiv:2501.04712]

Provides the mathematical foundation for our Pressing Accuracy signal (Signal 3):
- Time-to-intercept formula incorporating reaction time (τᵣ), max velocity (v_max), and direction penalty (τ_β)
- Logistic transform to compute interception probability per frame: `p_intercept = 1 / (1 + exp(-π/(√3·σ) · (T_limit - T_defender)))`
- Open-source implementation available via `unravelsports` Python package

**Key for our methodology:** We directly use Bekkers' time-to-intercept method to determine whether pressing decisions are correct (interception probable) or wasteful (interception unlikely). Our cognitive fatigue hypothesis: pressing accuracy degrades while physical capacity holds steady.

**Yagi et al. (2025)** — *Analysis of Line Break Prediction* [arXiv:2511.00121]
- XGBoost model for predicting defensive line breaks using 189 features from tracking + event data
- Found offensive player speed, defensive line gaps, and spatial distributions are key predictors (AUC = 0.982)
- Moderate correlation between line-break probability and shots/crosses conceded

**Relevance:** The spatial features (defensive line gaps, player distributions) used in Yagi et al.'s line-break model are similar to the spatial features we compute for Positional Drift (Signal 1). If fatigue increases line gaps, it should be detectable in both approaches.

**Karakuş & Arkadaş (2026)** — *Structural Pass Analysis in Football* [arXiv:2603.28916]
- Three structural metrics: Line Bypass Score, Space Gain Metric, Structural Disruption Index
- Combined into Tactical Impact Value (TIV)
- Identified four pass archetypes: circulatory, destabilising, line-breaking, space-expanding
- Build-up defenders are key drivers of structural progression

**Relevance:** The Space Gain Metric and Structural Disruption Index could serve as *outcome* measures for our fatigue model — does a fatigued defender allow more structurally disruptive passes?

### 2.3 Set-Piece & Specific Defensive Contexts

**Groom et al. (2026)** — as above, applied to corner kicks specifically.

**Bauer et al.** — CNN-LSTM for defender role assignment on corners (requires hand-labelled data).

**DeepMind/Liverpool (TacticAI)** — GNN for corner kick outcome prediction and suggestion generation, but not explicitly parameterised by interpretable tactical roles.

### 2.4 The Gap We Address

| What exists | What's missing |
|-------------|----------------|
| Groom et al.: defensive evaluation on corners | Open-play defensive fatigue |
| Bekkers: pressing intensity per frame, static τᵣ | Dynamic reaction time as a fatigue signal |
| Yagi et al.: line-break prediction | Whether fatigue *causes* increased line-break vulnerability |
| Thomas et al.: Quantile Cube for GPS load | Cognitive (not physical) fatigue detection from tracking data |
| DEFCON (Kim et al.): defensive credit within fixed organisation | How fatigue changes defensive organisation itself |

**Our contribution:** The first framework to detect cognitive fatigue in defenders by tracking behavioural degradation across multiple signals — while controlling for physical capacity.

---

## 3. Spatio-Temporal Methods — Toolbox for Our Pipeline

### 3.1 Formation & Position Identification

**Bekkers (2025)** — *EFPI: Elastic Formation and Position Identification* [arXiv:2506.23843]

Essential for Signal 1 (Positional Drift):
- Template matching using Hungarian algorithm across 65 predefined formations
- Scale-normalised assignment prevents illogical labelling
- Works per-frame or per-segment (5-min intervals align perfectly with our blocks)
- Open-source via `unravelsports` Python package

**Method:** We use EFPI to determine each defender's expected position per frame given team formation, then compute drift as deviation from that expected position.

### 3.2 Pitch Control & Spatial Models

**Spearman (2018)** — Pitch Control model (foundational reference)
- Computes each team's probability of controlling any point on the pitch
- Used in our Spatial Awareness signal (Signal 4) to compute danger values

**Pleuler's modification** — Direction penalty parameter (τ_θ) incorporated into time-to-intercept

### 3.3 Data Standards & Reliability

**Jo et al. (2026)** — *VERSA: Verified Event Data Format for Reliable Soccer Analytics* [arXiv:2601.21981]
- State-transition model for detecting anomalous event sequences
- Found 18.81% of K League events had logical inconsistencies
- VERSA cleaning significantly improved downstream VAEP performance

**Relevance:** We should apply similar sanity checks when merging our tracking and event data. Inconsistent event ordering could corrupt transition recognition timing (Signal 5).

**Yeung et al. (2025)** — *OpenSTARLab* [arXiv:2502.02785]
- Open-source framework for standardising event + tracking data into Unified Event Data / State-Action-Reward formats
- Includes deep learning event prediction and reinforcement learning tools

**Relevance:** OpenSTARLab's Pre-processing Package could save us time on data canonicalisation (pitch alignment, event synchronisation). Worth evaluating when data arrives.

### 3.4 Advanced Counterfactual Methods

**Kang & Narasimhan (2026)** — *Monte Carlo Pass Search* [arXiv:2606.11120] — CVPR 2026
- MCTS-based counterfactual pass evaluation using 3D ball trajectories
- World model generates hypothetical pass outcomes from sampled execution variants
- Distribution-aware attribution (mean-based and percentile-based scores)

**Relevance:** The percentile-based attribution approach could inform how we aggregate fatigue scores across multiple signals — not just mean degradation but distributional shifts in the tail.

---

## 4. Fatigue & Load Monitoring

### 4.1 Physical Fatigue in Football

**Thomas & Hannig (2025)** — *Movement Dynamics in Elite Female Soccer: The Quantile Cube Approach* [arXiv:2503.11815]
- Three-dimensional summary (velocity quantiles × acceleration quantiles × movement angle) per match half
- Significant 1st/2nd half differences in movement distributions
- Dirichlet-multinomial regression to identify positional and match-context effects
- Awarded 1st place at 2024 Carnegie Mellon Sports Analytics Conference

**Limitation for our work:** GPS-based external load (distance, sprint count) doesn't separate cognitive from physical fatigue. Our approach controls for physical metrics to isolate cognitive effects.

### 4.2 The Cognitive-Physical Distinction

Sports science literature consistently distinguishes:
- **Peripheral fatigue** — muscle contractile failure, reduced force output
- **Central fatigue** — reduced neural drive, impaired motor cortex output

Cognitive fatigue in team sports is linked to:
- Decision-making errors under high cognitive load (Smith et al., 2016)
- Reduced visual scanning and awareness (McGuckian et al., 2018)
- Increased reaction time in response to unpredictable stimuli (Coutinho et al., 2017)

**Our operationalisation:** We detect cognitive fatigue via the *dissociation* between behavioural degradation and physical capacity. If transition recognition slows (Signal 5) but recovery sprint speed holds steady, the deficit is cognitive, not physical.

### 4.3 The Substitution Validation Insight

Players substituted off typically show the highest fatigue accumulation. If our fatigue index spikes in the 15 minutes before substitution, that's strong convergent validity. This is a test we can run without needing experimental manipulation.

---

## 5. Methodological Context: How We Built on Prior Work

### 5.1 Model 1 — Pressure Exposure

Draws on:
- **Opponent proximity:** Direct from tracking data, calibrated using team-specific radii (c.f. StatsBomb's 4-5 yard rule)
- **Defensive depth:** Distance from own goal, mirroring defensive line analysis in Yagi et al. (2025)
- **Reorientation frequency:** Sharp heading changes >45° — adapted from movement intensity metrics in Thomas & Hannig (2025)
- **Transition count:** From event data, using possession-change detection

### 5.2 Signal 1 — Positional Drift

- **Formation template** via EFPI (Bekkers, 2025)
- **Expected position** from formation-aware assignment per frame
- **Ghosting concept** from Groom et al. (2026) — but applied to open-play, not set pieces

### 5.3 Signal 2 — Shift Latency

- **Dynamic τᵣ estimation** — Bekkers (2025) treats reaction time as a fixed parameter; we estimate it per defender per block
- **Trigger detection** from ball velocity spikes and opponent runs

### 5.4 Signal 3 — Pressing Accuracy

- **Time-to-intercept** formula from Bekkers (2025)
- **Active Pressing threshold** (speed < 2m/s → no pressing) also from Bekkers
- **Novel contribution:** Classifying presses as correct or wasteful based on interception probability

### 5.5 Signal 4 — Spatial Awareness

- **Pitch control** from Spearman (2018)
- **Ghost comparison** from Groom et al. (2026) — role-conditioned counterfactuals
- **Novel contribution:** Comparing actual spatial coverage vs. expected-position ghost coverage

### 5.6 Signal 5 — Transition Recognition Time

- **Transition detection** from event data (possession changes)
- **Recognition detection** from velocity vector direction changes
- **Cognitive-physical dissociation** — this is our most novel methodological contribution
- **Confound control:** Surprise vs. expected transitions from event context

---

## 6. Literature Gaps — Our Contribution

| Gap | How We Fill It |
|-----|---------------|
| **No cognitive fatigue framework for football defence** | First systematic framework with 5 behavioural signals |
| **Defensive analysis focuses on outcomes, not process** | We measure the process (positioning, timing, decisions) that precedes outcomes |
| **Fatigue studies use physical metrics only** | We isolate cognitive fatigue by controlling for physical capacity |
| **Most football analytics uses single data type** | We integrate tracking + event data for richer signal extraction |
| **Reaction time treated as fixed parameter** | We estimate τᵣ dynamically per defender per block as a fatigue indicator |

---

## 7. Key References

### Directly methodologically relevant

| Citation | Our use |
|----------|---------|
| Bekkers (2025) — Pressing Intensity [2501.04712] | Signal 3 methodology (time-to-intercept, pressing accuracy) |
| Bekkers (2025) — EFPI [2506.23843] | Signal 1 methodology (formation templates, expected positions) |
| Groom et al. (2026) — Off-Ball Defence [2601.00748] | Signal 4 inspiration (role-conditioned ghosting, defensive evaluation) |
| Spearman (2018) — Pitch Control | Signal 4 foundation (pitch control for danger values) |
| Yagi et al. (2025) — Line Break Prediction [2511.00121] | Spatial feature context, outcome validation |
| Thomas & Hannig (2025) — Quantile Cube [2503.11815] | Movement dynamics, quantile-based analysis |

### Useful context

| Citation | Value |
|----------|-------|
| Karakuş & Arkadaş (2026) — Pass Archetypes [2603.28916] | TIV as potential outcome measure |
| Jo et al. (2026) — VERSA [2601.21981] | Data quality framework |
| Yeung et al. (2025) — OpenSTARLab [2502.02785] | Data preprocessing tools |
| Kang & Narasimhan (2026) — MCPS [2606.11120] | Distribution-aware attribution ideas |
| Jiang et al. (2025) — GoalNet [2503.09737] | GNN-based evaluation context |

---

## 8. BibTeX File

See `references.bib` in the same directory.

---

*Compiled and synthesised: 24 Jun 2026 | Jervis 🧠*
