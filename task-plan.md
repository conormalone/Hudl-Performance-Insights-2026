# Focus Fatigue — Task Plan

**Last updated:** 24 Jun 2026 (v4 — split into subagent-ready chunks + decision points)

---

## How to Use This

- **Coding tasks** (🔧) are designed to be handed to a subagent: self-contained, clear inputs/outputs, hours-sized.
- **Decision points** (🧑‍⚖️) are where you need to weigh in. Marked with context on what's being decided.
- **Dependencies are explicit** — do tasks in numbered order within each section.

---

## 📦 FOUNDATION (Phase 0) — No Data Required

### F-0 Code Infrastructure — Setup

| # | Type | Task | Est. | Dependencies | Deliverable |
|---|------|------|------|-------------|-------------|
| F-0.1 | 🔧 | **Project skeleton** — `mkdir -p` structure (src/, tests/, data/, outputs/, notebooks/), `pyproject.toml` or `requirements.txt` with deps (pandas, numpy, scipy, matplotlib, kloppy, mplsoccer), git init + .gitignore | 1h | None | Runnable project skeleton ready for all downstream tasks |
| F-0.2 | 🧑‍⚖️ | **Confirm code structure conventions** — do you want notebooks for exploration vs. scripts for production? Any naming conventions? Docker or no? | 5min | F-0.1 | Agreed conventions |
| F-0.3 | 🔧 | **Tracking data loader stub** — `load_tracking(filepath) → pd.DataFrame` with columns: game_id, frame, player_id, x, y, team. Docstring only, no real data. | 1h | F-0.1 | `src/loaders/load_tracking.py` |
| F-0.4 | 🔧 | **Event data loader stub** — `load_events(filepath) → pd.DataFrame` with columns: game_id, frame, event_type, player_id, x, y, timestamp. Docstring only. | 1h | F-0.1 | `src/loaders/load_events.py` |
| F-0.5 | 🔧 | **Pitch visualisation** — `plot_frame(df, game_id, frame_id)` draws a standard pitch with player positions, colour-coded by team. Reuse throughout project. | 1.5h | F-0.1 | `src/viz/pitch.py` |
| F-0.6 | 🔧 | **Block segmenter** — `split_into_blocks(match_df, window_minutes=5) → list of DataFrames`. Frames per block, handles partial first/last blocks. Configurable window. | 1h | F-0.1 | `src/segments.py` |
| F-0.7 | 🔧 | **Baseline calculator** — `compute_baseline(player_df, first_n_minutes=15) → dict` of per-signal stats (mean, std). Handles rotation player fallback. | 1.5h | F-0.1 | `src/baselines.py` |

### F-0.5 Smoothing Pipeline

| # | Type | Task | Est. | Dependencies | Deliverable |
|---|------|------|------|-------------|-------------|
| F-0.5.1 | 🧑‍⚖️ | **Choose smoothing method** — Savitzky-Golay (simpler, fixed window) vs. Kalman filter (adaptive, more realistic). Refer to tracking quality if known, otherwise default to Savitzky-Golay. | 5min | None | Method decision |
| F-0.5.2 | 🔧 | **Implement Savitzky-Golay smoother** — `smooth_trajectory(df, window=7, polyorder=2) → df` with smoothed x, y, computed vx, vy columns. Per-player per-game. | 2h | F-0.5.1, F-0.1 | `src/smoothing.py` |
| F-0.5.3 | 🔧 | **Smoothing unit test** — verify on a known trajectory (straight line, step change): smoothed values are correct, velocity magnitudes are sensible. | 1h | F-0.5.2 | `tests/test_smoothing.py` |
| F-0.5.4 | 🔧 | **Hook smoothing into loader** — all subsequent loading pipelines automatically smooth on read. Single flag to disable. | 0.5h | F-0.5.2 | Updated `load_tracking.py` |

### F-1 Synthetic Data

| # | Type | Task | Est. | Dependencies | Deliverable |
|---|------|------|------|-------------|-------------|
| F-1.1 | 🧑‍⚖️ | **Define synthetic data parameters** — match length (90 min), frame rate (10fps or 25fps), noise level (5cm, 20cm), number of players, which defender should show fatigue vs. which stays fresh. | 10min | None | Parameter spec |
| F-1.2 | 🔧 | **Generate player trajectories** — 22 players with smooth curved paths (splines or Bézier). Include ball trajectory. Vary movement patterns: walk, jog, sprint, stand. | 3h | F-1.1, F-0.1 | `src/synthetic/generate_trajectories.py` |
| F-1.3 | 🔧 | **Add configurable noise** — apply Gaussian jitter to xy coordinates at configured level. Add occasional tracking dropout (missing frames). | 1h | F-1.2 | Noise layer in synthetic pipeline |
| F-1.4 | 🔧 | **Assign fatigue properties** — one defender has fatigue onset at minute 60 (increasing drift, slower reaction). Another stays fresh throughout. Non-defenders don't fatigue. | 1h | F-1.2 | `src/synthetic/fatigue_injector.py` |
| F-1.5 | 🔧 | **Generate synthetic event data** — basic possession changes, ball movements, periodic transitions. Synchronised to tracking frames. | 1.5h | F-1.2 | `src/synthetic/events.py` |
| F-1.6 | 🔧 | **Pipeline integration test** — load synthetic data through full pipeline (load → smooth → segment → baseline). Confirm no errors, sensible output shapes, no NaNs. | 1h | F-1.3–F-1.5, F-0.3–F-0.7 | `tests/test_pipeline_syntax.py` |

### F-2 Model 1: Pressure Exposure

| # | Type | Task | Est. | Dependencies | Deliverable |
|---|------|------|------|-------------|-------------|
| F-2.1 | 🧑‍⚖️ | **Set opponent proximity radius** — default 7m? Per-team calibration approach? Configurable parameter? | 5min | None | Radius decision |
| F-2.2 | 🔧 | **Opponent proximity** — per frame: count opponents within radius. Per block: mean count. | 1h | F-1.6, F-2.1 | `src/pressure/opponent_proximity.py` |
| F-2.3 | 🔧 | **Defensive depth** — per frame: distance from own goal line. Per block: mean + std. | 1h | F-1.6 | `src/pressure/defensive_depth.py` |
| F-2.4 | 🧑‍⚖️ | **Set reorientation threshold** — 45° in <1s is the default. Should this be tighter (60°) or looser (30°)? Depends on smoothing quality. | 5min | F-0.5 | Threshold decision |
| F-2.5 | 🔧 | **Reorientation frequency** — detect heading changes > threshold in <1s using smoothed velocities. Count per block. | 1.5h | F-1.6, F-2.4 | `src/pressure/reorientation.py` |
| F-2.6 | 🔧 | **Transition count** — count possession changes in defender's zone from event data, per block. | 1h | F-1.6 | `src/pressure/transition_count.py` |
| F-2.7 | 🔧 | **Weighted pressure composite** — combine 4 indicators: `pressure = 1 + Σ(indicator_i / baseline_i)`. Per block. | 1h | F-2.2, F-2.3, F-2.5, F-2.6 | `src/pressure/composite.py` |
| F-2.8 | 🔧 | **High/low pressure classification** — rank blocks, top quartile = high pressure, bottom quartile = low pressure (control). | 0.5h | F-2.7 | `src/pressure/classify.py` |

### F-3 Primary Signal Implementations

| # | Type | Task | Est. | Dependencies | Deliverable |
|---|------|------|------|-------------|-------------|
| F-3.1 | 🧑‍⚖️ | **Confirm signal implementation order** — recommended: Signal 5 (transition) first (highest value, simplest), then Signal 3 (pressing), then Signal 1 (drift), then Signal 2 (latency), then engagement metric. OK? | 5min | None | Priority agreement |
| — | — | — | — | — | — |
| F-3.2 | 🔧 | **Signal 5: Transition recognition** — detect turnover from event data. For each turnover, find first frame where smoothed defender velocity points toward own goal AND accelerating (speed increasing). Compute latency. Per-block mean + max. | 3h | F-1.6, F-0.5 | `src/signals/transition_recognition.py` |
| F-3.3 | 🔧 | **Signal 5 syntax check** — run on synthetic data: known-fatigue defender should show increasing latency post-minute 60. Known-fresh defender should not. | 0.5h | F-3.2 | `tests/test_signal5_basic.py` |
| — | — | — | — | — | — |
| F-3.4 | 🔧 | **Signal 3: Time-to-intercept (Bekkers)** — per defender-attacker pair per frame: compute `T = τᵣ + τ_dist + τ_β` using smoothed velocities. Convert to intercept probability via logistic. | 3h | F-1.6, F-0.5 | `src/signals/pressing/time_to_intercept.py` |
| F-3.5 | 🔧 | **Signal 3: Pressing event detection** — identify pressing actions: defender speed >2m/s AND moving toward an opponent. | 1h | F-3.4 | `src/signals/pressing/detect_presses.py` |
| F-3.6 | 🔧 | **Signal 3: Accuracy classification** — pressing event is "correct" if intercept_prob > 0.3, "wasteful" if below. Accuracy = correct / total per block. | 1.5h | F-3.5 | `src/signals/pressing/accuracy.py` |
| F-3.7 | 🔧 | **Signal 3 syntax check** — run on synthetic: accuracy should drop for fatigued defender, stay stable for fresh defender. | 0.5h | F-3.6 | `tests/test_signal3_basic.py` |
| — | — | — | — | — | — |
| F-3.8 | 🧑‍⚖️ | **EFPI template set** — which formations to include? mplsoccer's 65 default templates is standard. OK to use? | 5min | None | Template decision |
| F-3.9 | 🔧 | **Signal 1: Expected positions (EFPI-style)** — assign each defender to nearest template position per frame using Hungarian assignment. Scaled by team width/length. | 2h | F-1.6, F-3.8 | `src/signals/positional_drift/expected_position.py` |
| F-3.10 | 🔧 | **Signal 1: Drift computation** — `drift = ||actual_position - expected_position||` per frame. Per-block mean + 90th percentile. | 1h | F-3.9 | `src/signals/positional_drift/drift.py` |
| F-3.11 | 🔧 | **Signal 1 syntax check** — run on synthetic: drift should increase for fatigued defender in second half. | 0.5h | F-3.10 | `tests/test_signal1_basic.py` |
| — | — | — | — | — | — |
| F-3.12 | 🔧 | **Signal 2: Shift latency** — detect ball velocity spikes or opponent run triggers. For each trigger, measure time until defender velocity changes >30° toward the threat. Per-block mean + p90. | 3h | F-1.6, F-0.5 | `src/signals/shift_latency.py` |
| F-3.13 | 🔧 | **Signal 2 syntax check** — run on synthetic: latency should increase for fatigued defender, stay stable for fresh. | 0.5h | F-3.12 | `tests/test_signal2_basic.py` |
| — | — | — | — | — | — |
| F-3.14 | 🔧 | **Engagement/withdrawal metric** — per block: distance covered, number of directional changes, time spent sprinting, interventions per minute. Simple descriptive stats. | 2h | F-1.6, F-0.5 | `src/signals/engagement.py` |
| F-3.15 | 🔧 | **Engagement syntax check** — confirm metric produces sensible values for synthetic data. | 0.5h | F-3.14 | `tests/test_engagement_basic.py` |

### F-4 Exploratory Signals (Deferred)

| # | Type | Task | Est. | Dependencies | Deliverable |
|---|------|------|------|-------------|-------------|
| F-4.1 | 🔧 | **Signal 4: Spatial awareness** — simplified pitch control (Spearman). Compute opponent danger value at defender's actual location vs. expected location (from Signal 1). Gap = fatigue indicator. | 5h | F-3.9, F-1.6 | `src/signals/spatial_awareness.py` |
| F-4.2 | 🔧 | **Signal 4 redundancy check** — correlation with Signal 1. Also run on synthetic to confirm directionality. | 1h | F-4.1 | `tests/test_signal4_redundancy.py` |

### F-5 Dashboard & Tooling

| # | Type | Task | Est. | Dependencies | Deliverable |
|---|------|------|------|-------------|-------------|
| F-5.1 | 🔧 | **Per-signal time series plot** — x-axis = match minutes, y-axis = signal value, vertical lines = high-pressure blocks. Overlay fatigued vs. fresh defender. | 1.5h | F-3.x | `src/viz/signal_timeseries.py` |
| F-5.2 | 🔧 | **High vs. low pressure comparison plot** — boxplot or violin plot: signal value in high-pressure vs. low-pressure blocks, per signal. | 1h | F-3.x | `src/viz/pressure_comparison.py` |
| F-5.3 | 🔧 | **Pi compute benchmark** — generate 3-match synthetic dataset at realistic scale. Time: load → smooth → segment → compute 4 signals. Report per-game time. | 2h | F-1.6, F-3.x | Benchmark report |

---

## 📦 DATA-DEPENDENT (Phase 1-2)

### D-1 Data Arrival & Audit

| # | Type | Task | Est. | Dependencies | Deliverable |
|---|------|------|------|-------------|-------------|
| D-1.1 | 🧑‍⚖️ | **TRACKING QUALITY CHECK** — frame rate? Spatial noise level? Coordinate system (pitch-aligned or raw pixel)? Pitch dimensions? This decision determines the viability of all velocity-based signals. | 30min | Hudl data | Quality verdict |
| D-1.2 | 🔧 | **Load real tracking data** — run loader on Hudl files. Inspect schema, expected vs. actual columns, data types. | 1h | D-1.1 | Loaded DataFrames |
| D-1.3 | 🔧 | **Load event data** — load and synchronise with tracking frames by timestamp. Check alignment quality. | 1h | D-1.1 | Synchronised DataFrames |
| D-1.4 | 🧑‍⚖️ | **Decision: proceed or fallback?** — based on tracking quality. If <25fps or noise > 20cm, we lose velocity-dependent signals. Do we proceed with reduced signal set or trigger fallback? | 15min | D-1.1 | Go/no-go decision |
| D-1.5 | 🔧 | **Copy 3-match subset to USB** — select representative games (one high-intensity, one mid, one low). For Pi dev. | 0.5h | D-1.2 | USB staging |
| D-1.6 | 🔧 | **VERSA-style sanity checks** — event ordering, timestamps, missing events, logical consistency. | 1.5h | D-1.3 | Data quality report |
| D-1.7 | 🔧 | **Merge tracking + events** — unified per-frame dataset per match. | 2h | D-1.2, D-1.3 | Merged files |
| D-1.8 | 🧑‍⚖️ | **Player inclusion criteria** — minimum minutes? (30 min? 45 min?) How to handle substitutions (partial blocks, exclude, reset baseline)? | 10min | D-1.6 | Inclusion rules |
| D-1.9 | 🔧 | **Quick EDA notebook** — per-match overview, player minutes distribution, missing data, basic distributions of all fields. | 3h | D-1.7, D-1.8 | EDA report |

### D-2 Production Runs

| # | Type | Task | Est. | Dependencies | Deliverable |
|---|------|------|------|-------------|-------------|
| D-2.1 | 🔧 | **Apply smoothing to real data** — verify smoothing parameters work on real tracking (no over-smoothing, no artifacts). | 1h | D-1.7, F-0.5 | Smoothed match files |
| D-2.2 | 🔧 | **Run Model 1 (Pressure Exposure)** — all 4 indicators → weighted pressure → high/low classification. All 30 games. | 3h | D-2.1, F-2.x | `outputs/pressure_exposure/` |
| D-2.3 | 🔧 | **Run Signal 5** — transition recognition latency. All 30 games. | 2h | D-2.1, F-3.2 | `outputs/signals/signal5/` |
| D-2.4 | 🔧 | **Run Signal 3** — pressing accuracy. All 30 games. | 3h | D-2.1, F-3.6 | `outputs/signals/signal3/` |
| D-2.5 | 🔧 | **Run Signal 1** — positional drift. All 30 games. | 3h | D-2.1, F-3.10 | `outputs/signals/signal1/` |
| D-2.6 | 🔧 | **Run Signal 2** — shift latency. All 30 games. | 3h | D-2.1, F-3.12 | `outputs/signals/signal2/` |
| D-2.7 | 🔧 | **Run engagement metric** — distance, involvement, sprint count. All 30 games. | 2h | D-2.1, F-3.14 | `outputs/signals/engagement/` |
| D-2.8 | 🧑‍⚖️ | **Decision: run exploratory Signal 4?** — only if tracking quality is good AND Signal 4 shows non-redundancy with Signal 1. | 5min | D-2.5 | Go/no-go for Signal 4 |
| D-2.9 | 🔧 | **Run Signal 4 (exploratory)** — only if D-2.8 consensus. All 30 games. | 5h | D-2.8 | `outputs/signals/signal4/` |
| D-2.10 | 🔧 | **Aggregate all signals** — merge into unified fatigue dataset: per block × per player × per signal. | 1h | D-2.2–D-2.7 | `outputs/fatigue_dataset.parquet` |
| D-2.11 | 🔧 | **Compute composite fatigue index** — z-score each signal, weighted average (Signal 5 highest weight). | 1h | D-2.10 | `outputs/fatigue_index.parquet` |

### D-3 Validation & Analysis

| # | Type | Task | Est. | Dependencies | Deliverable |
|---|------|------|------|-------------|-------------|
| D-3.1 | 🔧 | **Convergent validity** — pairwise signal correlations in high-pressure blocks vs. low-pressure blocks. | 1.5h | D-2.10 | Correlation analysis |
| D-3.2 | 🔧 | **Discriminant validity** — correlate each signal with physical metrics (distance, sprint count). Pre-specified threshold: r < 0.2 = clean dissociation. | 1.5h | D-2.10 | Dissociation report |
| D-3.3 | 🔧 | **Temporal pattern analysis** — within-block trajectory (first 2min vs. last 2min of each block). Between-block recovery. Late-match attenuation (1st half vs. 2nd half). | 2h | D-2.10 | Temporal patterns |
| D-3.4 | 🔧 | **Substitution & reversal analysis** — (a) do signals spike 15 min before substitution? (b) after a high-pressure block, do signals return to baseline in subsequent low-pressure block? | 2h | D-2.10 | Substitution + recovery report |
| D-3.5 | 🧑‍⚖️ | **Decision: which signals make the paper?** — based on D-3.1–D-3.4 results. Which are clearly showing fatigue, which are noisy, which are redundant? | 30min | D-3.1–D-3.4 | Signal selection for paper |
| D-3.6 | 🔧 | **Sensitivity analysis** — vary: window size (3, 5, 7 min), pressure thresholds (top 20% vs. top 30%), opponent proximity radius (5m, 7m, 10m). Report effect size stability. | 4h | D-2.10 | Sensitivity report |
| D-3.7 | 🔧 | **Position-lane stratification** — wide defenders vs. central defenders. Exploratory only. | 2h | D-2.10 | Position-level report |
| D-3.8 | 🔧 | **Surprise vs. expected transitions** — flag transition types from event context. Re-run Signal 5 conditional on type. | 1.5h | D-2.3, D-1.7 | Context-conditional latency |
| D-3.9 | 🔧 | **Endogeneity checks** — (a) Model 1 from event data only (no tracking indicators), recheck correlations. (b) Frame blocking: even frames → pressure, odd frames → quality. (c) Placebo: shuffle pressure labels. | 3h | D-2.2, D-2.10 | Endogeneity report |
| D-3.10 | 🔧 | **Multiple comparison correction** — apply Benjamini-Hochberg to all signal × threshold × window tests. Report corrected significance. | 1h | D-3.6 | Corrected p-values |
| D-3.11 | 🧑‍⚖️ | **Decision: findings-driven paper or methodology paper?** — if signals pass validation (convergent + discriminant + temporal + endogeneity checks), we lead with findings. If mixed, we lead with framework. | 30min | D-3.1–D-3.10 | Narrative direction |

---

## 📦 RESULTS & PAPER (Phase 3-4)

| # | Type | Task | Est. | Dependencies | Deliverable |
|---|------|------|------|-------------|-------------|
| R-1 | 🔧 | **Publication-quality figures** — 4-6 figures: signal time series, high/low comparions maps, correlation heatmap, substitution plot. | 4h | D-3.x | Figure set |
| R-2 | 🔧 | **Results tables** — summary stats, effect sizes, corrected p-values, participant counts. | 3h | D-3.x | LaTeX/Markdown tables |
| R-3 | 🧑‍⚖️ | **Paper outline approval** — confirm section structure, lead narrative, which signals make the cut, title direction. | 30min | R-1, R-2 | Outline agreement |
| R-4 | 🔧 | **Write Introduction** — set up the problem: defensive analysis gap, cognitive-physical fatigue distinction, our approach. | 1d | R-3 | Draft section |
| R-5 | 🔧 | **Write Related Work** — synthesised from lit review doc. 3 subsections: defensive analysis, spatio-temporal methods, fatigue monitoring. | 1d | R-3 | Draft section |
| R-6 | 🔧 | **Write Methods** — Model 1, each primary signal, validation approach. Signal 4 in supplementary. | 2d | R-3, F-3.x, F-4.x | Draft section |
| R-7 | 🔧 | **Write Results** — narrative driven by figures and tables. | 2d | R-1, R-2, R-3 | Draft section |
| R-8 | 🔧 | **Write Discussion + Limitations** — interpretation, confounds, generalisability. | 1d | R-3 | Draft section |
| R-9 | 🔧 | **Full draft review** — internal consistency, flow, word count, formatting. | 1d | R-4–R-8 | Revision notes |
| R-10 | 🧑‍⚖️ | **Final paper review** — read entire draft, approve changes, sign off. | 1h | R-9 | Final manuscript |
| R-11 | 🔧 | **Presentation slides** — key visual narrative, abstract, 3-4 key figures, implications. | 2d | R-1, R-10 | Slide deck |

---

## 🔄 FALLBACK (If tracking data is unusable)

| # | Type | Task | Est. | Dependencies | Deliverable |
|---|------|------|------|-------------|-------------|
| X-1 | 🔧 | **Download SkillCorner public dataset** — 10 games, check schema compatibility. | 2h | None | Fallback data ready |
| X-2 | 🧑‍⚖️ | **Decision: proceed with fallback?** — commit to SkillCorner or StatsBomb-only route. | 15min | D-1.4 or X-1 | Fallback direction |
| X-3 | 🔧 | **StatsBomb 360 — Signal 5 only** — transition recognition from event data (no tracking needed). | 2h | X-2 | Reduced analysis |
| X-4 | 🔧 | **StatsBomb 360 — Signal 3 (pressing proxy)** — approximate pressing from event context (no tracking = no Bekkers TTI). | 2h | X-2 | Reduced analysis |
| X-5 | 🔧 | **Write Plan B abstract** — 3 sentences validating the methodology paper narrative. | 30min | X-2 | Plan B abstract |
| X-6 | 🔧 | **Repivot paper outline** — cut tracking-dependent sections, lead with method + event-only signals. | 1h | X-5 | Revised outline |

---

## 📋 Decision Points Summary (Quick Reference)

| When | Decision | What's at stake |
|------|----------|-----------------|
| Before anything | Code structure conventions (F-0.2) | Minor — workflow preference |
| Before smoothing (F-0.5.1) | Smoothing method (Savitzky-Golay vs. Kalman) | Affects all velocity signals downstream |
| Before synthetic gen (F-1.1) | Synthetic data parameters (noise, frame rate) | Only affects dev syntax-checking |
| Before pressure (F-2.1, F-2.4) | Opponent radius + reorientation threshold | Tuning parameters — sensitivity test covers this |
| Before signals (F-3.1) | Signal implementation order | Priority, not irreversible |
| Before EFPI (F-3.8) | Formation template set | Minor — 65 is standard |
| **When data arrives (D-1.1)** | **Tracking quality verdict** | **Most critical decision — determines signal viability** |
| After quality check (D-1.4) | Proceed or fallback? | Affects entire Phase 1-2 plan |
| After EDA (D-1.8) | Player inclusion criteria | Affects sample size and results |
| After real data runs (D-2.8) | Run exploratory Signal 4? | Low priority, affects one signal |
| **After validation (D-3.5, D-3.11)** | **Which signals make the paper? Findings or methodology paper?** | **Defines the paper contribution** |
| Before writing (R-3) | Paper outline + narrative direction | Sets the writing scope |
| End (R-10) | Final paper sign-off | Publishing-ready |

---

## Current Status

```
F-0.1  ⬜ Project skeleton
...
```

Doned at the existing task plan already had tasks broken down into small chunks. But I think he wants them split even further into the smallest possible units - the kind of thing you could give to a coding subagent and say "do this one thing." The plan now has ~70 tasks, each small and completable.
