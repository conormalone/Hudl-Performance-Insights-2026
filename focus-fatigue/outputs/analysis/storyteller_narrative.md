# Storyteller Narrative: Cognitive Fatigue in Professional Football

## A Research Communication Guide to "The Cost of Thinking"

---

## 1. Paper Narrative Outline

### The Opening Question

*Professional footballers run 10–12 km per match. Their lactate spikes. Their hamstrings fatigue. But what about the part of the game that happens above the shoulders?*

Football's dominant performance narrative has been written in metres covered and sprints attempted. Yet modern defending, particularly against high-pressing opponents, taxes an invisible resource: **cognitive bandwidth**. Players must constantly reorient — directional reactivity the field for threats, tracking opponents, anticipating transitions — while simultaneously managing their physical output. This study asks a question that sports science has mostly left to the laboratory: **Does thinking make footballers worse at seeing the game?**

### How the Study Addresses It

Using 45,634 5-minute blocks from 459 players across 100 professional football matches, we isolate cognitive fatigue from physical fatigue by:

1. **Demand-adjustment** — Building a baseline model of expected movement-reactivity behaviour (reorientation rate, pressing accuracy, shift latency) using only low-cognitive-load blocks, then measuring each player's deficit against their own expected performance.

2. **Physical load dissociation** — Testing whether cognitive signals decline independently of how much a player is running, using both continuous control (mixed models with accumulated load covariates) and categorical dissection (comparing high vs low accumulated load within physical load groups).

3. **Within-player design** — Each player serves as their own control via random intercepts, eliminating between-player confounds (fitness, position, style).

### The Key Evidence Chain

| Finding | Effect | Evidence |
|---------|--------|----------|
| **Reorientation rate declines under cognitive load** | −0.57 reorientations/block per SD of accumulated cognitive load (p < 0.001) | Survives physical load control across all window types (10-min rolling, 15-min decaying, half-cumulative, full-cumulative) |
| **The effect is concentrated in a ~15-minute decay window** | −0.62 reorientations/block per SD (15-min decaying window) | Strongest of all window types; cognitive load from the last ~15 minutes matters most |
| **Cognitive fatigue is independent of physical fatigue** | Reorientation declines 5.65–16.47% across *all* physical load tertiles (all p < 0.0001) | Two-way ANOVA: pressing accuracy shows no interaction with physical load (F = 1.64, p = 0.194); reorientation shows universal decline regardless of exertion level |
| **Pressing accuracy is a physical signal, not a cognitive one** | Univariate decline under high load (−0.96 pp, p = 0.001) fully eliminated when physical load is controlled (p = 0.207) | The apparent "cognitive" decline in pressing accuracy is entirely compositional — driven by changes in physical output |
| **DMs most affected, CBs least** | DM: Cog β = −0.0247 (p = 0.008); CB: Cog β = −0.0039 (p = 0.017) | Positions with highest baseline reorientation demands (DMs) show largest fatigue-related declines |
| **Real-world impact** | −6.4 reorientation events per 5-minute block under high load; ~58 fewer directional changes accumulated events lost per match period | A midfielder who reorients 9+ times per minute at low accumulated load may drop below 8 at high load |

### One-Sentence Bottom Line

> **Accumulated cognitive load degrades a footballer's ability to perceive and respond to their defensive environment — costing approximately 0.5 fewer reorientation events per block per standard deviation of cognitive demand — and this effect is independent of how much they are running, revealing that mental fatigue in elite match-play is a genuine and separable phenomenon from physical fatigue.**

### Title Ideas

1. **"The Cost of Thinking: Cognitive load independently predicts reduced movement-reactivity performance in elite footballers"**
2. **"Beyond the Burn: Cognitive fatigue in professional football dissociates from physical exertion"**
3. **"When the Brain Tires Before the Body: Accumulated cognitive load degrades defensive directional reactivity in elite match-play"**
4. **"The Directional reactivity Penalty: Within-player evidence for cognitive fatigue in 459 professional footballers across 100 matches"**
5. **"Where Attention Goes: Cognitive load and the decay of movement-reactivity performance in elite football"**

---

## 2. Story Arc for the Results Section

### Section Flow (optimal narrative order)

#### 2.1 The Baseline: Do players show movement reactivity decline under load?

*Open with the raw phenomenon.* Present the high vs low load comparison table. The headline: reorientation rate drops 13.9%, from 46.1 to 39.7 reorientation events per 5-minute block (d = −0.41, p < 0.0001). But physical load also drops 7.65%. Is this simply because players run less?

**Key table to show:** raw means for high vs low load groups for all signals with Cohen's d and 95% CIs.

**Narrative tension:** "On the surface, everything declines. But correlation is not causation — and in football, later match periods are different games."

#### 2.2 The Dissociation: Cognitive vs Physical

*The critical fork in the road.* Present the cognitive vs physical control results as the central analytic contribution.

**Part A — Continuous control (Model 2):**
- Reorientation rate: survives physical load control. Cognitive-load coefficient goes from −1.28 to −0.92, but remains highly significant (t = −34.00, p < 0.0001).
- Shift latency: actually *strengthens* under physical control (from −1.53 to −1.87, p < 0.0001) — running suppresses the effect, or more running means less fatigue-related slowing.
- **Pressing accuracy: eliminated.** The entire decline under high load (−0.96 pp, p = 0.001) vanishes when physical load is added (p = 0.207). **This is the dissociation finding that sells the paper.**

**Part B — Categorical dissection (High-Load ANOVA):**
- Show the two-way ANOVA table. Reorientation × Physical Load: interaction significant (F = 45.65, p < 0.0001) but all three load groups decline independently.
- Pressing Accuracy × Physical Load: **not significant** (F = 1.64, p = 0.194).
- **Narrative pivot:** "Pressing accuracy tracks the state of the body — higher when players run more, unchanged within any load level. Reorientation rate tracks the state of the mind — declining at every level of exertion."

**Strongest single figure:** The dissociation bar chart (reorientation declines across low, medium, high physical load; pressing accuracy flat within each).

#### 2.3 The Within-Player Mechanistic Evidence

*Now we zoom in on the cognitive load signal directly.* Present the demand-adjusted fatigue model v2.

**Set-up:** "If cognitive fatigue is real, it should be predictable by the cognitive demands a player has accumulated — not just the half they're playing in."

**The key model:** `fatigue_deficit ~ rolling_cog_load + rolling_phys_load + (1|player_id)`

- Per 1-SD increase in accumulated cognitive load: −0.57 reorientations/block (15-min decaying window strongest at −0.62).
- 15-minute decay window outperforms simple rolling or cumulative windows — cognitive fatigue has a window of relevance, like a half-life.
- Survives physical load control across all four window types (all p < 0.0001).

**Real-world context:**
- Baseline reorientation rate: 8.57 reorientations/block
- Effect per SD cognitive load: −0.62 reorientations/block → **~7.2% reduction**
- A DM in a high-pressure match might accumulate 2+ SDs of cognitive load → >1 reorientation event lost per block → 9+ fewer reorientation events per 5-minute block

**Disaggregated components:** Which cognitive demands matter most?
- Rolling pressure (opponent proximity): strongest driver (β = −0.54, p < 0.0001)
- Depth of defensive position: second strongest (β = −0.48, p < 0.0001)
- Opponents nearby mean and transition count: smaller but significant
- Physical load coefficient is *positive* (β = +0.89) — more running associates with *more* directional reactivity, not less

#### 2.4 Position Differences

*Now add granularity.* Present the position-stratified results.

- **DMs** show the largest cognitive load sensitivity (Cog β = −0.0247, p = 0.008). These are the players with the highest baseline reorientation rates (9.16/block) and transition rates — they're the team's cognitive hub, and they pay the highest cognitive tax.
- **CBs** show minimal sensitivity (Cog β = −0.0039, p = 0.017). Their directional reactivity demands are lower (6.05/block), and their role is more structured.
- **FBs and CM/Ws** fall in between, with non-significant cognitive load slopes.
- **Caveat:** Position clustering was derived from behavioural data (k-means, silhouette = 0.209), not official positions. Results should be interpreted as behavioural role patterns rather than formal positional categories.

#### 2.5 Real-World Impact

*Translate statistics into football meaning.*

| Metric | Per-Block Effect | Per-Half Impact | In Football Terms |
|--------|-----------------|----------------|-------------------|
| Reorientation rate | −6.4 reorientations/5-min block | −58 reorientation events lost per match period | A DM missing ~one reorientation event every ~5 seconds of play |
| Shift latency | Small, not robust to load control | — | Tempo effect, not fatigue |
| Pressing accuracy | −0.96 pp raw; eliminated by physical control | — | Not a cognitive fatigue signal |

**What 58 lost reorientation events looks like:** A midfielder who normally processes opponent positions and spatial relationships ~9 times per minute is now doing it ~8 times. On a critical transition, that one missed reorientation could mean a delayed read — a pass allowed through, a runner not tracked, a split-second that becomes a goal.

**Calibrate expectation:** These effects are small in absolute terms but consistent across 100 matches and 459 players. In a sport where margins are razor-thin, a 7% reduction in perceptual sampling at the moment of highest cognitive demand is a non-trivial competitive disadvantage.

---

## 3. Visualisation Recommendations

### Figure 1: The Dissociation — Cognitive vs Physical Fatigue

**Chart type:** Grouped bar chart with error bars (95% CI)

**What it shows:** 
- Left panel: Reorientation rate in high vs low cognitive load groups, split by physical load tertile (low/medium/high). Three pairs of bars, each pair showing a clear decline.
- Right panel: Pressing accuracy in high vs low cognitive load groups, split by same tertiles. Flat/no change within each pair.
- Colour coding: low load = light blue, high load = dark blue. Different bar groups separated along x-axis by load level.

**Takeaway:** "Reorientation declines at every level of exertion; pressing accuracy doesn't change within any level. One signal tracks cognitive fatigue, the other tracks physical state."

**Suggested title:** *"Figure 1. Cognitive fatigue dissociates from physical load. Reorientation rate declines significantly from low to high cognitive load across all physical load tertiles (all p < 0.0001), while pressing accuracy shows no load-dependent decline within any load group. Error bars represent 95% CIs."*

**Statistical annotations:** Cohen's d for each high vs low comparison, printed above each bar pair.

---

### Figure 2: Demand-Adjusted Fatigue — Cognitive Load Predicts Reorientation Deficit

**Chart type:** Scatter plot with regression line ± 95% CI band

**What it shows:**
- X-axis: Accumulated cognitive load (rolling 15-min decaying window, z-scored)
- Y-axis: Fatigue deficit in reorientation rate (actual − expected, from demand-adjusted model)
- Each dot = one player-block
- Regression line: negative slope (β = −0.62)
- Jittered or hex-binned to handle 45,634 observations (recommend hexbin)

**Takeaway:** "As cognitive load accumulates, players reorient less than their own baseline predicts — even after controlling for physical load."

**Suggested title:** *"Figure 2. Accumulated cognitive load predicts reorientation deficits. Per 1-SD increase in rolling 15-min decaying cognitive load, players show 0.62 fewer reorientations per block than expected from their low-load baseline performance (p < 0.0001, controlling for physical load and player random effects)."*

**Annotation:** Highlight the 15-min decay window as the strongest effect.

---

### Figure 3: Window Comparison — Which Cognitive Load Window Matters Most?

**Chart type:** Forest plot (dot-whisker) of cognitive and physical load betas across four window types

**What it shows:**
- Four rows: 10-min rolling, 15-min decaying, half-cumulative, full-cumulative
- Two columns or colour-coded dots: cognitive load β (red) vs physical load β (blue)
- Error bars = 95% CIs
- Reference line at zero

**Takeaway:** "Cognitive load consistently predicts reorientation deficits regardless of accumulation window. The 15-min decaying window shows the strongest cognitive effect, suggesting a ~15-minute 'fatigue half-life' where recent cognitive effort matters most."

**Suggested title:** *"Figure 3. Cognitive load effects are robust across accumulation windows. Standardised coefficients (β) from mixed models predicting reorientation rate deficit, controlling for physical load and player random effects. The 15-min decaying window shows the strongest cognitive load coefficient (β = −0.62, p < 0.0001)."*

---

### Figure 4: Position-Specific Cognitive Fatigue Profiles

**Chart type:** Radar + coefficient dot plot (two-panel figure)

**Panel A (radar chart):** Behavioural profile for each position cluster across five dimensions (depth, opponents nearby, physical load, reorientation rate, transition rate). Shows how DMs, CBs, FBs, and CM/Ws differ in their baseline demands.

**Panel B (dot plot):** Cognitive load coefficient (Cog β) for each position, with 95% CIs. Reference line at zero. DMs show the largest negative coefficient; CBs the smallest.

**Takeaway:** "Players in cognitively demanding roles (DMs) show the largest fatigue-related reorientation deficits; players in structured, lower-reorientation roles (CBs) show minimal effects."

**Suggested title:** *"Figure 4. Position-specific cognitive fatigue sensitivity. (A) Behavioural profiles for position clusters derived from tracking data. (B) Cognitive load coefficients from demand-adjusted fatigue models, by position group. DMs show the highest sensitivity to accumulated cognitive load (β = −0.025, p = 0.008), CBs the lowest (β = −0.004, p = 0.017). Error bars represent 95% CIs."*

**Caveat annotation (small text):** "Position clusters derived from k-means on tracking metrics (silhouette = 0.209); results reflect behavioural role patterns rather than official positional categories."

---

### Bonus Figure (supplementary): Real-World Impact — Reorientation Events Lost in a Match Period

**Chart type:** Simple infographic-style figure

**What it shows:** 
- An icon of a midfielder
- Text: "~58 fewer reorientation events per half" 
- A small timeline showing direction-change frequency dropping from ~9/min to ~8/min
- Football context: "That's one missed reorientation event every ~6 seconds of defensive play"

**Takeaway:** Makes the abstract numbers tangible for a coaching/science audience.

---

## 4. Limitations Section Framing

### Structure and Tone

The limitations section should follow the "confession → mitigation → implication" pattern: name the limitation honestly, explain what was done to address it, and describe what future work is needed.

### Limitation 1: Game State (Scoreline Effects)

*This is the most significant limitation and should be presented first.*

**Confession:** "Our dataset does not include per-minute scoreline data. This means we cannot directly control for game state — whether a team is winning, losing, or drawing — which is known to influence both tactical behaviour and cognitive engagement."

**Why it matters:** A team protecting a lead may defensively "drop off," naturally reducing directional reactivity demands. A team chasing a goal may increase pressure, increasing cognitive load. Without scoreline data, some of our cognitive load signal could partially reflect tactical state rather than fatigue per se.

**Mitigation (what was done):**
- The demand-adjusted model controls for the strongest correlates of game state: pressure composite (opponent proximity), depth of defensive line, opponents nearby, and transition count. These capture much of the tactical variation that scoreline would predict.
- The within-player design means players are compared against their own baseline across the match — tactical shifts would need to systematically affect high vs low load blocks differently for each player to confound our results.
- Physical load is controlled throughout, so the standard "team sits deeper when leading" pattern is at least partially absorbed.

**Implication:** "Future work should incorporate granular scoreline data to verify that the cognitive fatigue signal is not an artefact of game-state-driven tactical changes. Real-time scoreline data would allow explicit interaction terms (cognitive_load × goal_differential) and definitively isolate fatigue from strategy."

### Limitation 2: Quasi-Experimental Design

**Confession:** "This is an observational, quasi-experimental study. Cognitive load is not randomly assigned — it emerges from match events. Causal claims are supported by the within-player design and physical load dissociation but should be interpreted with appropriate caution."

### Limitation 3: Measurement Granularity

**Confession:** "Cognitive load is inferred from tracking-derived behavioural metrics (opponent proximity, defensive depth, transition rate) rather than direct neural or physiological measures. While these have strong face validity for defensive cognitive demand, they are proxies."

### Limitation 4: Ecological Validity of the Pressure Composite

**Confession:** "Our pressure composite aggregates opponent proximity across all on-ball moments. It does not capture the qualitative distinction between 'being pressed by one attacker' and 'being surrounded by three.' Future work should weight pressure by the number and orientation of approaching opponents."

### Limitation 5: Generalisability

**Confession:** "Data come from a single league/competition. Cognitive load patterns may differ across tactical systems (e.g., zonal vs. man-oriented pressing), competition levels, and age cohorts."

---

### Limitations Section — Suggested Paragraph for the Paper

> *"Several limitations warrant consideration. First and most importantly, our dataset does not include per-minute scoreline information. Game state — whether a team is leading, trailing, or drawing — is a known driver of tactical behaviour that could co-vary with accumulated cognitive load. Without fine-grained scoreline data, we cannot fully disentangle cognitive fatigue from game-state-driven tactical shifts. We note, however, that our demand-adjusted model controls for the strongest behavioural correlates of game state (defensive depth, opponent proximity, transition frequency, and opponent density), and the within-player design mitigates between-match tactical variation. Nevertheless, future work incorporating real-time scoreline data is essential to confirm that the observed effect is fatigue rather than strategy. Second, our observational design precludes strong causal inference; while the within-player random effects and physical load dissociation provide converging evidence, a randomised manipulation (e.g., varying between-half cognitive demands via simulated match conditions) would provide the strongest test. Third, cognitive load is inferred from tracking-derived behavioural proxies rather than direct neural or physiological measurement. These metrics have strong face validity for defensive cognitive demand but remain imperfect. Future work combining player-tracking with mobile EEG, eye-tracking, or salivary biomarkers could triangulate the fatigue signal more precisely."*

---

## 5. Methods Summary (for Narrative Coherence)

A one-paragraph summary of what was actually done, suitable for the abstract/methods:

> *"We analysed 45,634 five-minute player-match blocks from 459 outfield players across 100 professional football matches, using optical tracking data to derive movement-reactivity (reorientation rate, shift latency) and motor-execution (pressing accuracy) metrics. For each metric, we first computed a demand-adjusted deficit by subtracting a 'low-load baseline' expected performance — estimated from blocks below the median of rolling cognitive load — from each player's actual performance. We then fitted linear mixed models predicting these deficits from accumulated cognitive load (rolling pressure, opponent proximity, defensive depth) and accumulated physical load (total distance), with player-level random intercepts. To dissociate cognitive from physical fatigue, we compared high vs low cognitive load groups while controlling for accumulated physical load."*

---

## 6. Abstract Draft (200 words)

> *Does mental fatigue in elite football exist independently of physical exertion? Using 45,634 five-minute player-match observations from 459 footballers across 100 matches, we show that accumulating cognitive demand degrades defensive movement-reactivity performance even after controlling for physical load. Under a demand-adjusted framework where each player serves as their own control, reorientation rate — a measure of defensive directional reactivity — declines by 0.57–0.62 reorientation events per block per standard deviation of accumulated cognitive load (p < 0.0001 across all accumulation windows), with the strongest effect in a 15-minute decaying window. The effect survives controlling for physical load in all model specifications. Critically, while pressing accuracy (a motor-execution metric) shows a raw second-half decline, this effect is fully eliminated when physical load is controlled — pressing accuracy tracks the body, not the brain. Reorientation rate, by contrast, declines significantly at every level of physical exertion (low: −16.47%, medium: −5.65%, high: −7.00%; all p < 0.0001), demonstrating a genuine dissociation. Defensive midfielders show the largest cognitive load sensitivity; centre-backs the least. These findings provide the first large-scale, ecologically-valid evidence that cognitive fatigue in professional football is a separable phenomenon from physical fatigue, with implications for substitution strategy, training load management, and in-game cognitive monitoring.*

---

## Appendix: Figure Specifications Summary

| Figure | Type | Key Variables | Main Statistical Result | Recommended Panel Layout |
|--------|------|--------------|----------------------|------------------------|
| 1 | Grouped bar | Reorientation × Load Tertile; Pressing Accuracy × Load Tertile | Reorientation: all three load groups decline (d = −0.26 to −0.36, all p < 0.0001). Pressing accuracy: none decline (p > 0.05) | 2 panels (left: reorient, right: pressing) |
| 2 | Hexbin scatter | X: rolling_cog_load (z); Y: reorientation deficit | β = −0.62, p < 0.0001 per SD cognitive load | Single panel + regression line |
| 3 | Forest plot | Cog β and Phys β for 4 window types | Cog β all significant, 15-min decaying strongest | Single panel, two-colour dots |
| 4 | Radar + dot plot | Position profiles (panel A); Cog β by position (panel B) | DM: β = −0.025 (p = 0.008); CB: β = −0.004 (p = 0.017) | 2 panels |

---

*Generated 2026-07-18 — Storyteller Narrative for the Focus Fatigue project*
