# Focus Fatigue — Work Plan (v3)

**Date:** 10 Jul 2026
**Data source:** Stats Perform optical tracking (25fps) — 100 matches
**Paper target:** Journal/conference (StatsBomb 2026 rejected — alternate venue TBD)
**Deadline:** ~1 November 2026 (~16 weeks)

---

## Data Overview

| Source | Scope | Details |
|--------|-------|---------|
| **Tracking** (`tracking.parquet`) | 100 matches | 25fps, all 22 players + ball, full match, 159 MB/match |
| **Shapes** (`shape.json`) | 100 matches | Formation templates every 60s for both teams, in & out of possession, with role positions and fit scores, 774 KB/match |
| **Total** | 100 matches | ~16 GB — fits comfortably on USB (28.8 GB available) |

### Key differences from v2 plan

| v2 (broadcast + StatsBomb) | v3 (Stats Perform) |
|----------------------------|-------------------|
| 30 games broadcast tracking (2-4fps) | **100 matches optical tracking (25fps)** |
| 5 seasons StatsBomb 360 event data | **No event data** — tracking-derived possession only |
| No pre-computed formations | **Pre-computed shapes every 60s** — EFPI step eliminated |
| Cloud compute likely needed | **Pi + USB handles everything** (16 GB / 28.8 GB) |
| `goalkeeper` flag expected | **GKs not flagged** — identify via jersey #1 + shape.json role descriptions |
| Single ID system across data | **Two separate ID systems** — bridge via jersey + team + time window |

### Shape.json — the hidden superpower

The pre-computed `inPossession`/`outOfPossession` formations with per-role `averageRolePositionX/Y` and `fitScore` give us "expected position" baselines **for free**. No need for EFPI clustering or Hungarian assignment — the expected position for a defender in a given minute is just their role's `averageRolePosition` from the shape entry.

This collapses the implementation plan:
- **Signal 1 (positional drift):** `drift = ||player_xy - shape_role_position||` — three lines of code
- **Signal 4 (spatial awareness):** compare actual position vs shape role position — simpler than pitch control model

### What we lose without event data

| Lost | Replacement |
|------|-------------|
| Event-level transition timestamps | `team_in_possession` field in tracking — flip detection gives turnover frames |
| Event context (surprise vs. expected transitions) | Heuristic based on ball speed before/after flip (fast ball = surprise) |
| Shot/xG data for validation | Not needed for core fatigue signals |

---

## Phase 0 — Foundation Work (finalise)
### Jul 10 – Jul 13 | ~3 days

| Day | Task |
|-----|------|
| **Today** | **Update plan for new data (done)**, audit 1 match schema end-to-end, map columns to signals |
| Jul 11 | Finalise data directory structure for 100 matches, write schema mapper |
| Jul 12 | Seed data: copy 3 matches to USB for development. Validate venv + deps. |
| Jul 13 | **Go/no-go checkpoint** — confirm all 5 signals are computable from tracking alone |

**Deliverable:** Working schema mapper + 3-match dev dataset on USB

---

## Phase 1 — Data Ingestion & Signal Implementation
### Jul 14 – Aug 3 | ~3 weeks

| Week | Task |
|------|------|
| **W1** | **Data ingestion pipeline** — bulk loader for 100 matches (tracking + shapes), canonicalise coordinates, normalise DOP (all attack left→right), frame deduplication checks. |
| **W2** | **Model 1 (Pressure Exposure)** — opponent proximity, defensive depth, reorientation frequency, transition flips. Baseline construction per player. Block classifier. |
| **W3** | **Model 2 signals (batch 1)** — Signal 5 (transition recognition) from `team_in_possession` flips. Signal 3 (pressing accuracy) via Bekkers TTI. |
| **Stop-loss** | By Aug 3, cumulative dev time < 60 hours OR first 3 matches fully processed. |

**Deliverable:** Ingestion pipeline + Model 1 + Signal 5 + Signal 3 on 3-match sample

### Signal implementation order (recommended)

1. **Signal 5 → Transition recognition** ← highest value, simplest
   - `team_in_possession` flip → detect defender reorientation → latency per transition
   - Cognitive-physical dissociation: compare latency vs recovery sprint speed
2. **Signal 3 → Pressing accuracy** ← Bekkers method from papers
   - Time-to-intercept per defender-attacker-frame pair
   - Accuracy = correct presses / total presses per block
3. **Signal 1 → Positional drift** ← shape.json makes this trivial
   - Match player to shape role via jersey + team + minute window
   - `drift = ||pos - role_centroid||` per frame
4. **Signal 2 → Shift latency** ← most complex trigger detection
   - Ball speed spikes or opponent runs → defender reaction time
5. **Signal 4 → Spatial awareness** ← exploratory, lowest priority

---

## Phase 2 — Full 100-Match Production Run
### Aug 4 – Sep 7 | ~5 weeks

| Week | Task |
|------|------|
| **W4** | Bulk run Model 1 on all 100 matches — pressure exposure, block classification |
| **W5** | Bulk run Signal 5 + Signal 3 on all 100 matches |
| **W6** | Bulk run Signal 1 + Signal 2 on all 100 matches |
| **W7** | Aggregate all signals → unified fatigue dataset. Run validation checks |
| **W8** | Iteration — sensitivity analysis, parameter tuning, robustness tests |
| **Stop-loss** | Sep 7: all core signals computed on all 100 matches. Full dataset ready. |

**Deliverable:** Unified fatigue dataset (`outputs/fatigue_dataset.parquet`) + validation report

---

## Phase 3 — Results & Validation
### Sep 8 – Sep 28 | ~3 weeks

| Week | Task |
|------|------|
| **W9** | Convergent validity (signals agree on high-pressure blocks). Discriminant validity (signals ≠ running metrics). |
| **W10** | Substitution validation, temporal pattern analysis, position-level stratification |
| **W11** | Sensitivity analysis (window sizes, thresholds), endogeneity checks, multiple comparison correction |
| **Deliverable** | Final results: which signals pass validation, effect sizes, publication-ready figures |

**Note:** No writing in this phase. Results drive the paper narrative, not the other way around.

---

## Buffer Week
### Sep 29 – Oct 5 | ~1 week

- On schedule → extra polishing, deeper validation, or start writing early
- Behind → catch-up without eating into writing time

---

## Phase 4 — Paper Writing
### Oct 6 – Oct 26 | ~3 weeks

| Week | Task |
|------|------|
| W13 | Introduction + Related Work + Methods |
| W14 | Results + Discussion + Limitations |
| W15 | Full draft review, revisions, polish, formatting |

**Deliverable:** Complete manuscript ready for submission
**Venue:** TBD (journal rather than conference given the extended timeline)

---

## Phase 5 — Final Polish
### Oct 27 – Nov 1 | ~6 days

- Publication-quality figures final pass
- Supplementary materials (appendix, reproducibility instructions)
- Cover letter / submission metadata

**Deliverables:** Submission-ready paper + any presentation materials

---

## Key Changes from v2 Plan

| v2 assumption | v3 reality | Impact |
|--------------|------------|--------|
| 30 matches broadcast + 5 seasons StatsBomb | 100 matches Stats Perform tracking | More data, less variety (no event context) |
| EFPI clustering for expected positions | Shape.json provides them pre-computed | Eliminates F-3.8, F-3.9 — saves ~3h |
| Event data for transition timestamps | `team_in_possession` in tracking | Works fine, but can't classify transition type |
| Cloud compute may be needed | Pi + USB handles everything | No cloud costs, simpler stack |
| Goalkeeper flag in data | GKs not flagged (all `goalkeeper=False`) | Must filter by jersey #1 + shape role |
| `player_id` cross-references cleanly | Different ID systems (ints vs strings) | Must bridge via jersey + team + time window |

---

## Critical Dependencies

| Dependency | Drop-dead date | Fallback |
|------------|---------------|----------|
| 100 matches accessible on USB | Jul 13 | Proceed with whatever is available (even 30 is enough) |
| Shape.json schema consistent across all 100 | Jul 20 | Validate on first 10; if inconsistent, fall back to tracking-only (loss of Signal 1 shortcut) |
| Pi can process 1 match in <15 min | Jul 20 | If slower, batch process overnight / reduce to 50 matches |
| At least 2 signals pass discriminant validity | Sep 28 | Pivot to methodology paper (framework contribution + descriptive stats) |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Shape.json schema differs across 100 matches | Low | Medium | Validate on first 10 matches; if inconsistent, EFPI fallback (compute formations from tracking) |
| Pi processing too slow for 100 matches | Low-Medium | Medium | Start with 30 matches; can always add more. Batch overnight. |
| `team_in_possession` too noisy for transition detection | Low | High | Validate on 3-match sample in Phase 1; if noisy, use ball speed heuristic instead |
| Null fatigue signal | Medium | Medium | Lead with framework/methodology; 5-signal battery is itself a contribution |
| No event data limits story (no xG, no pass sequences) | Medium | Low | Focus on tracking-only signals; cite limitation and call for integrated data |
| GB limit on USB (28.8 GB, 16 GB data + overhead) | Low | Low | Process matches sequentially, delete raw parquet after processing |

---

## Key Milestones

1. **Jul 13** — Schema confirmed, 3-match dev dataset ready, column-to-signal mapping frozen
2. **Jul 27** — Model 1 running on 3 matches, baselines working
3. **Aug 3** — First 2 signals (Signal 5 + Signal 3) validated on 3 matches
4. **Sep 7** — All signals computed on all 100 matches
5. **Sep 28** — Validation complete, signals that make the cut confirmed
6. **Oct 20** — Full paper draft
7. **Oct 26** — Final manuscript
8. **Nov 1** — Submission ready

---

*Created: 10 Jul 2026 | Jervis 🧠*
