# Peer Review: Focus Fatigue — Cognitive Fatigue in Football Defence

**Reviewer:** Subagent (automated peer review)
**Date:** 24 June 2026
**Documents reviewed:**
- `work-plan.md` (v2, 21 Jun)
- `methodology-defensive-quality.md` (draft, 21 Jun)
- `task-plan.md` (24 Jun)

**Severity key:**
- 🔴 **BLOCKING** — Must fix before Phase 1
- 🟡 **CONCERN** — Should address; risk of invalid results
- 🔵 **NITPICK** — Minor; consider for polish

---

## A. OVERALL APPROACH

### A1. The Two-Model Architecture — Sound in Concept, Fragile in Execution

**Rating:** Good idea, underwhelming implementation plan.

Separating Pressure Exposure (Model 1) from Defensive Quality (Model 2) is the right conceptual decomposition. The arrow `Model 1 → Model 2` (context → response) is logical. However, the current framing has a **structural endogeneity problem**:

> `methodology.md` Lines 11–15: Both models draw from the **same tracking data** (x,y coordinates). The 4 Load indicators and 5 Signal indicators all use the same raw inputs.

If a defender's positional drift (Signal 1) and opponent proximity (Load indicator 1) are both computed from the same tracking frames, then a correlation between high pressure and positional drift may simply reflect that both variables are computed from the same source data. This isn't a test of cognitive fatigue — it's a test of whether two functions of the same data co-vary.

**🟡 CONCERN — Endogeneity:** The architecture needs a clearer story about why the correlation between Model 1 and Model 2 outputs is not a mechanical artifact of shared data inputs. Consider: using a subset of tracking frames for exposure and a non-overlapping subset for quality; or constructing Model 1 from event data only and Model 2 from tracking.

### A2. The 5 Signals — Some Collapse, Some Missing

**Claimed independence is overstated.** The five signals likely reduce to ~3 constructs:

| Construct | Signals | Overlap Concern |
|-----------|---------|-----------------|
| **Positional quality** | Signal 1 (Drift) + Signal 4 (Spatial Awareness) | Signal 4 literally uses Signal 1's expected position as its counterfactual. They are not independent. Signal 4 is Signal 1². |
| **Reaction time** | Signal 2 (Shift Latency) + Signal 5 (Transition Recognition) | Both measure time-to-respond to an external event. The cognitive mechanism is nearly identical; the triggers differ (ball movement vs. possession change). |
| **Decision quality** | Signal 3 (Pressing Accuracy) | This is actually the most distinct signal. |

**🔵 NITPICK:** If you're reporting 5 separate signals, readers will ask about overlap. You should plan to show inter-signal correlations and justify keeping all 5. A composite index (Option B) partly solves this, but then you lose the granularity you tout as a strength.

**Missing signals worth considering:**

| Missing Signal | Why It Matters |
|----------------|----------------|
| **Ball-playing quality under pressure** | Do fatigued defenders make worse passes? Turnover rate when in possession is a direct cognitive load proxy. |
| **Coordination breakdown** | Do defenders drift out of sync *with each other*? Pairwise distance variance between CBs across high-pressure blocks. |
| **Effort/engagement withdrawal** | Does the defender reduce involvement? Tracking distance covered, interventions attempted per minute. This is a classic fatigue response — conserving energy by doing less. |
| **Anticipation error** | Does the defender get caught ball-watching? Measured as time-delay between opponent's attacking movement and defender's first response. |

**🟡 CONCERN — Coverage gap:** The 5 signals all measure *quality of execution*, but none measures *quantity of engagement*. Cognitive fatigue can manifest as doing less (withdrawal) just as much as doing worse (degradation). You need at least one involvement/engagement metric.

### A3. Validation Strategy — Not Convincing Enough

From `methodology.md` Lines 130-140:

> **1. Convergent evidence:** Do multiple signals point in the same direction per block?

As noted above, if signals share data sources, convergence is expected mechanically. This is not strong validation.

> **2. Discriminant validity:** Do the signals *not* correlate with raw running metrics?

**🟡 CONCERN:** This is the best test in the list, but it has a flaw: if a player is genuinely fatigued (both cognitive and physical), the signals *should* correlate with running metrics. A null result here could mean either (a) you found cognitive-specific fatigue (the paper's claim) or (b) your signals are too noisy to detect anything. You need an a priori effect size threshold — what counts as "does not correlate"? r < 0.2? r < 0.1?

> **3. Temporal pattern:** Does fatigue increase monotonically across match halves within high-pressure blocks?

**🔴 BLOCKING assumption:** Fatigue is **not** monotonically increasing — it's episodic, it ebbs and flows. Players recover during low-intensity periods (that's the whole point of the two-model architecture). You need to define what temporal pattern *would* constitute fatigue evidence. I'd suggest: (a) fatigue score increases during high-pressure blocks but decreases or stabilises during low-pressure blocks, and (b) this within-block pattern attenuates as the match progresses (less recovery capacity late in game).

> **4. Substitution validation:** For players substituted off, do our signals spike in the 15 minutes before substitution?

🟡 CONCERN — See Section C4 below.

**Missing validation elements:**
- **No cross-validation or train/test split** anywhere. With 30 games (~60 defender-match observations), you could do leave-one-game-out or k-fold.
- **No multiple comparison correction.** You're testing 5+ signals × multiple pressure thresholds × multiple time windows. At α=0.05, you'll get false positives by chance. Report whether findings survive Bonferroni or Benjamini-Hochberg correction.
- **No pre-registration or analysis plan.** If this is confirmatory (testing specific hypotheses), the hypotheses should be registered before seeing the data. If it's exploratory, say so explicitly.

### A4. Weakest Part of the Approach

**🔴 Signal 4 (Spatial Awareness) — by a wide margin.**

This signal requires:
1. Formation inference (EFPI clustering) → error source 1
2. Expected position estimation → error source 2
3. A pitch control model (Spearman) → error source 3
4. A "ghost" comparison (actual vs. expected danger value) → error source 4

Four layers of assumptions, each with measurement error, cascading into the final metric. The ghost comparison multiplies errors from Signals 1 and 4. This signal will produce noise, not signal.

**Recommendation:** Cut Signal 4 from the primary analysis. Run it as an exploratory sensitivity check. Lead with Signals 5 and 3 (cleanest implementations, clearest cognitive-physical distinction). That's still a strong 3-signal paper (or 4 if you keep drift and latency separate).

---

## B. TASK PLAN QUALITY

### B1. Task Sizing — Mostly Good

Tasks are 1–5 hours, which is the right granularity. Exceptions:

| Task | Estimate | Concern |
|------|----------|---------|
| F-1.1 (Synthetic generator) | 2h | **🔴 Too small.** This is supposed to "validate everything that follows" (your words). A 2-hour synthetic data generator will produce toy data with unrealistic noise properties. Realistic synthetic data needs: noise models (jitter, occlusion), realistic player movement (smooth curves, not random walks), realistic pressure distributions, player identity swaps, missing frame patterns. Budget 6-8h or adjust expectation. |
| F-2.2 (Pressing accuracy, Bekkers) | 4h | **🟡 Ambitious.** Bekkers' model is non-trivial (time-to-intercept, direction penalty, logistic intercept probability). Without reusable code from the paper, this is a 6-10h implementation. Do you have permission to use their code? Is it publicly available? |
| F-2.5 (Spatial awareness) | 4h | **🟡 Undersold.** Simplified Spearman pitch control + formation inference + ghost comparison in 4h? This is a 2-3 day task for a single developer. Recommend cutting this entirely (see A4) or re-estimating at 16-20h. |

### B2. Missing Tasks

**Critical omissions — tasks that don't exist but should:**

| Missing Task | Why Critical | Source Doc |
|-------------|--------------|------------|
| **Pi compute benchmark** | Work-plan Phase 0 promises it. No task exists. If the Pi can't process 30 games × tracking at reasonable speed, you need the cloud fallback *before* Phase 1. | `work-plan.md` "Phase 0" |
| **Cloud storage setup** | Work-plan says 27GB is tight on Pi. Where's the AWS/cloud bucket setup task? | `work-plan.md` "Storage note" |
| **Tracking noise filtering / smoothing** | Broadcast tracking is noisy. No task to implement a Kalman filter, Savitzky-Golay smoother, or occlusion handler. Without this, velocity-based metrics will be garbage. | `methodology.md` "Open Questions Q5" |
| **Statistical testing framework** | No task for building the hypothesis testing infrastructure (effect sizes, confidence intervals, correction for multiple comparisons). You'll need this for Phase 2. | `task-plan.md` D-3.x |
| **Player inclusion criteria** | What's the minimum minutes threshold to include a player in the analysis? 30 min? 45 min? This should be defined and coded before seeing the data. | `methodology.md` (implied) |
| **Substitution handling in segmentation** | When a player is subbed on/off, how do their blocks work? Partial blocks? Excluded blocks? Reset baseline? | `methodology.md` "Open Questions Q2" |
| **Pre-registration / analysis plan document** | If this work aspires to be more than exploratory, the hypotheses and analytic plan should be frozen before data arrives. | — |
| **Reproducibility audit (for analyses)** | R-3 mentions re-running everything — good. But you also need a task for: pinning dependency versions, containerisation (Docker?), random seed fixing. | `task-plan.md` R-3 |

### B3. Ordering / Dependency Issues

| Issue | Detail | Severity |
|-------|--------|----------|
| **D-1.6 (USB copy) is last in D-1 block** | You should copy the subset to USB *before* running EDA on the Pi. The order should be: D-1.1/1.2 (load and inspect), D-1.3 (quick quality check), D-1.6 (copy to USB), then D-1.4/1.5 (full merge + EDA on the laptop). As written, you'd do all processing on the Pi first, then copy. | 🟡 CONCERN |
| **No dependency link between synthetic validation and real-data runs** | F-3.1 (test signals on synthetic) should be a gating requirement for D-2.1 (run Model 1 on real data). If signals fail on synthetic data, you shouldn't touch real data. The task plan doesn't make this explicit. | 🟡 CONCERN |
| **F-2.x order vs. methodology recommendation** | Methodology says "Signal 5 first" then "Signal 3". Task plan has F-2.1 (Signal 5) and F-2.2 (Signal 3) — correct. But F-2.4 (Signal 2, Shift Latency) is listed before F-2.5 (Signal 4). The methodology doesn't say, but Signal 2 depends on accurate velocity estimation which depends on tracking quality — should validate tracking quality first. | 🔵 NITPICK |

### B4. Synthetic Data Strategy — Useful but Overconfident

**🟡 CONCERN — The synthetic data approach is underspecified.**

F-1.1 at 2h is meant to "generate dummy tracking frames for a single match with known properties". The document says it "validates everything that follows." This is dangerously optimistic.

A synthetic generator that validates the full pipeline needs:
- Realistic noise characteristics (broadcast tracking has ~10-50cm jitter, identity swaps, occlusion)
- Realistic player movement (not random walks — smooth curved paths with accelerations)
- Realistic pressure distributions (clustered by match phase, not uniform)
- Multiple defenders with known fatigue properties (one fatigues, one doesn't)

**Without realism in the synthetic data, passing tests on synthetic data provides near-zero confidence for real data.**

**Recommendation:** Either (a) budget 6-8h for a realistic synthetic generator, or (b) reframe synthetic data as "syntax checking" — verifying the pipeline runs without errors — not "validation." Then rely on the real-data validation tasks (D-3.x) for actual correctness.

### B5. Fallback Plan — Adequate but Thin

The fallback plan is structurally reasonable:

> SkillCorner 10-game public dataset → same pipeline
> StatsBomb 360 → reduced (3 of 5 signals lost)
> Repivot to methodology paper

**🟡 CONCERN — Risks not fleshed out:**

| Risk | Detail |
|------|--------|
| **SkillCorner ≠ Hudl schema** | No task to check schema compatibility *before* Phase 1. If SkillCorner uses different coordinate systems, frame rates, or player identifiers, you lose time. |
| **StatsBomb 360 → only Signals 3 and 5** | A 2-signal paper is much weaker. "Defensive quality" with only pressing accuracy + transition recognition is a significant scope reduction. Does Plan B account for this narrative shift? |
| **"Methodology paper" is vague** | What's the methodological contribution? "Here's how to compute things on public data" is not novel. Successful methodology papers introduce new methods. If this is the Plan B, the contribution should be articulated now. |

**Recommendation:** Before Phase 1, validate that SkillCorner's public data schema is compatible with your pipeline. Write a concrete Plan B abstract. If the Plan B for a methodology paper can't be articulated in 3 sentences, it's not a real plan.

---

## C. SPECIFIC CONCERNS

### C1. 5-Minute Blocks vs. Adaptive Windows

**Rating:** 🟡 CONCERN — Ecologically weak but pragmatically defensible.

**Arguments for adaptive windows:**
- Game phases (settled defence, transition, attack, dead ball) are the natural units of cognitive load
- A 5-min block that spans 3 phases conflates different cognitive states
- An adaptive approach (e.g., "cluster frames by game state, then aggregate within each state") would give cleaner signal

**Arguments for 5-min blocks (why they might still win):**
- Pragmatic simplicity — fixed windows are trivial to implement and compare
- Work-plan mentions sensitivity analysis for 3-min/7-min; if results are robust, the concern is mostly theoretical
- Adaptive windows introduce their own boundary problem (where does one "phase" end?)
- Many fatigue studies in team sports use fixed windows (10-min epochs, quarters, halves)

**🔴 But the sensitivity analysis is too narrow.** Testing 3-min, 5-min, and 7-min doesn't address the phase boundary problem. A 3-min block that starts at 42:30 might split the pre-half defensive stand + half-time differently than one that starts at 41:00. This is a **block alignment artifact**.

**Recommendation:** Keep 5-min blocks as primary, but add a **phase-conditional variant** as a robustness check. For each block, label the dominant game state (settled defence, transition, dead ball, opponent possession, own possession). Then re-run analysis controlling for phase composition of each block. If results survive this control, they're much more convincing.

### C2. Broadcast Tracking Quality — Potentially Blocking

**Rating:** 🔴 BLOCKING — This is the biggest risk in the project.

The methodology relies heavily on velocity-based metrics:

| Signal | Velocity Dependency |
|--------|-------------------|
| Reorientation frequency (Model 1) | 45° heading change in <1s → needs velocity vector at ~10fps minimum |
| Shift Latency (Signal 2) | Time from trigger to "velocity vector changes >30° toward threat" |
| Transition Recognition (Signal 5) | "First frame where defender velocity vector points toward own goal AND accelerating" |
| Pressing Accuracy (Signal 3) | Speed >2m/s → pressing event classification depends on velocity magnitude |

**The problem:** Broadcast tracking (Second Spectrum, Opta) operates at ~10-25 fps with significant positional noise. At 10 fps:
- Velocity vectors from consecutive frames have ~5-15° of directional noise from pixel jitter alone
- Acceleration estimation (needed for Signal 5's "defender is accelerating" check) requires double-differencing, which amplifies noise ~3-5x
- The 45° reorientation threshold in <1s is right at the noise floor — many "reorientations" will be tracking jitter

**What you need to check before Phase 1:**
1. What is the actual frame rate of Hudl's broadcast tracking data? (10fps? 25fps?)
2. What is the spatial noise level? (1-2 pixels? 10-20cm?)
3. Are there pitch-aligned coordinates or raw pixel coordinates?
4. Does the data include player identity confidence scores?

**If tracking quality is poor, you lose Signals 1, 2, and arguably 4 and 5.** You'd be left with Signal 3 (Pressing Accuracy, which uses a simplified velocity threshold) and a degraded version of the transition recognition.

**Until this is verified, the entire velocity-based methodology is contingent.**

### C3. Centre-Back vs. Full-Back Stratification

**Rating:** 🔵 NITPICK — Worth doing, but explicitly frame as exploratory.

The question is reasonable: CBs and FBs have different physical and cognitive demands. FBs cover more distance, engage in more transitions, face more 1v1 situations. CBs manage the defensive line, track runners, organise shape.

**But with 30 games:**
- Each game has ~2 CBs and ~2 FBs (assuming 4-4-2 or 4-3-3)
- That's ~60 CB-match and ~60 FB-match observations
- After filtering for minimum minutes and position consistency, you might have 50 and 40 observations
- This is enough for a secondary analysis, not a primary finding

**Concern:** The methodology lists this as "Open Question Q3" but doesn't specify whether it's primary or exploratory. If it appears in the paper's main findings, the sample size needs justification.

**🟡 CONCERN — The real position confound is not CB vs. FB.** It's **wide defenders vs. narrow defenders in different formations.** A FB in a 4-4-2 has different demands than a wing-back in a 3-5-2. You might need to stratify by formation *and* position, which reduces your per-cell sample to ~10-15 observations. At that point, you're running subgroup analyses on noise.

**Recommendation:** Stratify by position-lane (wide vs. central) as a sensitivity check. Present it as "exploratory evidence of differential fatigue profiles" at most. Do not lead with it.

### C4. Substitution Validation — Weaker Than Advertised

**Rating:** 🟡 CONCERN — Useful supporting evidence, not "validation."

From `methodology.md`:
> For players substituted off, do our signals spike in the 15 minutes before substitution?

**Confounds that undermine causal inference:**

| Confound | Problem |
|----------|---------|
| **Selection bias** | Players aren't substituted randomly. They're subbed off for: poor performance, injury, tactical reasons, disciplinary reasons, or time-wasting. Poor performance *is* correlated with our fatigue signals. The substitution validation may simply show that "bad defensive performance → substitution" — which is true by definition. |
| **Injury confound** | If a player is subbed off injured, any pre-sub signal spike could be pain/deceleration, not cognitive fatigue. |
| **Tactical substitution** | If a team is chasing a game, the manager may sub off a CB for a forward. The CB wasn't fatigued — they were tactically sacrificed. Your signal would falsely flag fatigue. |
| **Sample size** | How many defender substitutions in 30 games? Defenders are subbed less frequently than attackers (≈2-3x less). You might have ~15 defender substitution events. With 15 observations and multiple confounds, meaningful statistical inference is impossible. |
| **Direction of causation** | Signal spikes → substitution, OR impending substitution → signal spikes? The latter is plausible: a player who knows they're coming off may reduce effort or lose focus. This would be a behavioural confound, not a fatigue signal. |

**What would be stronger:**

- **Natural experiment:** Unplanned substitutions (e.g., a teammate gets a red card → CB stays on) vs. planned ones
- **Reversal test:** Do players who play a full 90 after a high-pressure block show recovery (signals return to baseline)?
- **Between-player comparison:** Do players who play 90 minutes in a high-pressure game show different fatigue trajectories than those who play 90 in a low-pressure game?

**Recommendation:** Keep the substitution analysis but frame it as descriptive/supportive, not validating. The "substitution validation" heading implies a level of causal identification that the study design cannot support.

---

## SUMMARY OF PRIORITY ISSUES

### 🔴 BLOCKING (Fix before Phase 1)

| # | Issue | Location | Recommendation |
|---|-------|----------|---------------|
| 1 | **Tracking quality for velocity-based metrics** is unverified | `methodology.md` Q5, Signals 2/5 | **Must** verify frame rate and spatial noise of Hudl broadcast tracking. If <25fps or >20cm noise, redesign velocity-dependent signals. |
| 2 | **Spatial Awareness (Signal 4)** has 4 cascading assumption layers | `methodology.md` §4, `task-plan.md` F-2.5 | Cut from primary analysis. Make exploratory only. Or re-estimate at 16-20h. |
| 3 | **Synthetic data generator is 2h for a "validates everything" claim** | `task-plan.md` F-1.1 | Either budget 6-8h for realistic synthetic data, or reframe as syntax-checking only. |
| 4 | **No tracking noise filtering task exists** | `task-plan.md` | Add a task: Kalman filter or smoothing for tracking coordinates. |
| 5 | **Validation lacks multiple comparison correction or cross-validation** | `methodology.md` §Validation | Add Bonferroni/BH correction and train/test split to validation plan. |
| 6 | **Pi compute benchmark is promised but has no task** | `work-plan.md` Phase 0, `task-plan.md` | Add task for benchmarking Pi performance on tracking data. |

### 🟡 CONCERNS (Should address)

| # | Issue | Location | Recommendation |
|---|-------|----------|---------------|
| 7 | **Endogeneity: both models share tracking data inputs** | `methodology.md` Architecture | Construct Model 1 from events-only as robustness check, or use non-overlapping frame subsets. |
| 8 | **5 signals collapse to ~3 constructs** | `methodology.md` §2 | Plan to show inter-signal correlations. Consider reducing to 3 primary signals. |
| 9 | **No engagement/withdrawal metric** | `methodology.md` §2 | Add a metric for effort withdrawal (distance/min, interventions/min). |
| 10 | **"Monotonic fatigue" assumption is wrong** | `methodology.md` Validation #3 | Redefine temporal pattern: within-block increase, between-block recovery, late-match attenuation. |
| 11 | **Substitution validation has selection bias** | `methodology.md` Validation #4 | Reframe as descriptive. Add reversal test (post-sub recovery). |
| 12 | **Pressing Accuracy task is under-estimated** | `task-plan.md` F-2.2 | Budget 6-10h or confirm Bekkers code availability. |
| 13 | **USB copy task out of order** | `task-plan.md` D-1.6 | Move earlier in D-1 sequence. |
| 14 | **No explicit quality gate before real-data runs** | `task-plan.md` F-3.x → D-2.x | Add: "F-3.x tests pass → proceed to D-2" as a dependency. |
| 15 | **SkillCorner schema compatibility unverified** | `task-plan.md` X-1 | Add a task to validate schema before Phase 1. |

### 🔵 NITPICKS

| # | Issue | Location | Recommendation |
|---|-------|----------|---------------|
| 16 | CB vs. FB stratification is fine but small sample for conclusions | `methodology.md` Q3 | Frame as exploratory secondary analysis. |
| 17 | Block alignment artifact not addressed | `methodology.md` | Add phase-conditional robustness check. |
| 18 | No pre-registration or hypothesis freezing | `task-plan.md` | Pre-register if this is confirmatory. |
| 19 | "Methodology paper" Plan B is vague | `task-plan.md` X-3 | Write a concrete 3-sentence abstract now. |
| 20 | Plan B for StatsBomb 360 loses 3/5 signals | `task-plan.md` X-2 | Articulate the 2-signal paper narrative now. |

---

## FINAL VERDICT

**The project has a strong core idea and a well-structured plan overall.** The two-model decomposition is smart, the focus on transition recognition is the right instinct, and the task granularity is appropriate for the timeline.

**However, four things worry me:**

1. **Tracking quality is the elephant in the room.** If Hudl's broadcast tracking is standard quality (10fps, noisy), the velocity-based metrics collapse. This should be verified *before* investing in the full infrastructure. (🔴 Blocking)

2. **Signal 4 (Spatial Awareness) is over-engineered.** It layers too many assumptions. Cut it, lead with Signals 5 and 3, and run Spatial Awareness as an exploratory add-on. (🔴 Blocking)

3. **The synthetic data pipeline is dangerously optimistic.** A 2-hour generator won't test anything meaningful on real data. Either invest real effort or be honest that it's only a syntax check. (🔴 Blocking)

4. **The validation strategy has gaps.** No multiple comparison correction, no cross-validation, monotonicity assumption is wrong, substitution validation has confounds. These need addressing. (🟡 Concerns)

**If you fix the 6 blocking issues, this is a solid project with a realistic chance of producing a meaningful findings-driven paper.**

---

*Review generated 24 Jun 2026 | Peer review v1*
