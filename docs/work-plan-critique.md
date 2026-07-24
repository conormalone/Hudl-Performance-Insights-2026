# Work Plan Critique — Hudl Performance Insights 2026

**Reviewer:** Jervis (AI Research Assistant)
**Date:** 21 June 2026

---

## Executive Summary

The plan is well-structured and shows sensible thinking about parallelisation and dependencies. However, it has several significant issues — the most critical being an overly optimistic view of what one person + one AI assistant on a Raspberry Pi can deliver in 15.5 weeks, especially when the data is entirely unseen. Below is a structured critique covering each dimension requested.

---

## 1. Timeline: Is 15.5 Weeks Realistic?

**Barely, and only with luck on your side.**

The math works on paper, but there's zero slack. Every phase is back-to-back with no buffer. Real research never goes to plan — data is messier than expected, null results require rethinking, pipelines break, the Raspberry Pi runs out of storage mid-analysis. A single two-week delay in Phase 1 or 2 cascades through the entire schedule, leaving Phase 4 (writing) or Phase 5 (presentation) squeezed.

**Specific timeline concerns:**

- **Phase 1 (2.5 weeks):** Data acquisition + schema audit + pipeline setup + EDA is at least 3–4 weeks of work for one person on unknown data. If the schema is undocumented or the data is large (Hudl tracking data can be gigabytes per match), cleaning alone could take a week. On a Raspberry Pi with 27 GB of usable storage, a dataset of even moderate size will force painful tradeoffs (subsampling, aggressive compression, partial analysis). Two and a half weeks is aggressive.
- **Phase 2 (4 weeks):** Feature engineering + statistical modelling + advanced analysis + robustness checks is genuinely 4–6 weeks of work, especially if the analysis path is exploratory and iterative (which it will be, because the data is unknown). This phase is the core intellectual contribution — rushing it risks shallow analysis.
- **Phase 5 (1 week):** A submission-ready slide deck, abstract, and polished paper in one week is unrealistic. Presentations are deceptive time sinks — good ones take 2–3 rounds of revision. This needs 2 weeks minimum.

---

## 2. Phase Scoping: Effort Assessment

### Phase 1 — Data Acquisition & Exploration (2.5 weeks)
*Under-scoped.* This is the most dangerous phase because the data is unknown. If the raw data is large (multiple GB), the Pi's storage becomes an immediate constraint. Cleaning, schema discovery, and ETL on a Pi's ARM processor will be noticeably slower than on a laptop. I recommend budgeting 3–4 weeks and starting ETL prototyping with synthetic data in June.

### Phase 2 — Core Analysis (4 weeks)
*Slightly under-scoped.* Feature engineering from raw tracking/event data is itself 1–2 weeks of work. Then statistical modelling, clustering/ML (compute-bound on Pi), and robustness checks. Realistically 5 weeks for quality work, 4 weeks if you're focused and the data cooperates.

### Phase 3 — Results & Validation (4 weeks)
*Structurally confused.* This phase bundles three distinct activities:
1. Final figures/tables (pre-writing)
2. Peer review / sanity checks (validation)
3. Methods section draft (writing)

The methods draft belongs in Phase 4. Putting it here moots the "spend October on writing" goal — you've already started writing in September. This phase should be split or refocused entirely on validation and figures, with methods moved to Phase 4.

### Phase 4 — Paper Writing (4 weeks)
*Barely adequate.* Writing a complete academic manuscript from scratch in 4 weeks while simultaneously interpreting final results is tight even for experienced writers. The milestone claiming a full draft by Oct 15 (day 17 of a 28-day phase) is inconsistent with the phase timeline. Either the milestone is wrong, or the scope is impossible.

### Phase 5 — Presentation (1 week)
*Under-scoped.* 7 days for slides + abstract + submission formatting + final proofreading. Realistically 2 weeks.

---

## 3. Risk Assessment: What's Missing

The plan identifies four risks, which is a good start. Here's what's missing:

### Missing Risk 1: Raspberry Pi Compute & Storage Constraints
This is the elephant in the room. 27 GB of usable USB storage is barely enough for a single modern tracking dataset, let alone intermediate files, models, figures, and code. Factor of 10 risk: the dataset doesn't fit. There's no fallback for remote compute (e.g., cloud VM, university cluster) mentioned. The plan needs:
- A hard storage budget (max dataset size, aggressive cleanup between phases)
- A cloud/rented-compute fallback explicitly planned
- Benchmarking early — test load/process time on Day 1, not Week 4

### Missing Risk 2: Null or Ambiguous Results
The plan mentions this as "Analysis path unclear, results null/ambiguous" but doesn't plan for it. What happens if the statistical models show nothing interesting? There's no contingency for pivoting to descriptive analysis, different hypotheses, or a different narrative framing. This should be a "Plan B" sketched out before data arrives.

### Missing Risk 3: Single-Person Bottleneck
One human + one AI assistant means no parallel human work. If Conor gets sick, has a work crunch, or loses motivation mid-project, there's zero redundancy. The plan should assume Conor has limited availability (e.g., evenings/weekends if this is a side project) and plan accordingly.

### Missing Risk 4: Literature Review Is Unplanned
The plan says "Literature review can start immediately" — but there's no concrete deliverable, no deadline, no reading list, no synthesis document. This is a classic "we'll get to it" item that will end up rushed into Phase 4. A half-baked literature review weakens the paper significantly.

### Missing Risk 5: Format Uncertainty
"Assume standard academic format" is the fallback for unknown conference/journal specs, but "standard" varies wildly (short papers, long papers, specific section requirements). If the target venue has strict page limits or format requirements discovered late, rewriting is painful.

### Missing Risk 6: Reproducibility Debt
Phase 3 includes "confirm all results reproducible" but doing this retroactively is painful. Reproducibility should be built in from Phase 1 (seeded random numbers, pinned library versions, Makefile or equivalent pipeline).

---

## 4. The "Spend Most of October on Writing" Goal

**This goal is achievable only if Phase 3 is restructured.**

As written, Phase 3 includes a methods draft (which is writing) during September. And Phase 4 starts October at the earliest. So you get *most of October* for writing, which is exactly 4 weeks. Possible, but not comfortable.

**To make this work:**
- Phase 3 must produce *all final results and figures* by Sep 28, full stop. No methods writing in Phase 3.
- Everything needed for writing (tables, figure drafts, statistical output) must be in a handoff document by Sep 28.
- Phase 4 then becomes the "write the paper" phase entirely. 4 weeks for a full paper is doable but tight — Conor should expect to write 500+ words/day on writing days.

**Bigger concern:** If Phase 3 runs over (because validation surfaces issues or figures need multiple revisions), October writing time disappears. The current plan has no mechanism to detect or prevent this.

---

## 5. Structural Issues

1. **No explicit buffer.** Every phase ends when the next begins. In real projects, things slide. Build in at least one "buffer" week (perhaps between Phase 3 and 4) as schedule insurance.

2. **Phase 3 is a grab bag.** Sanity checks + figures + methods + supplementary materials + reproducibility is too many things. Split it: Phase 3a (figures + validation), Phase 3b (methods draft, reproducibility). Or move methods to Phase 4 entirely.

3. **Milestone inconsistency.** Milestone 5 says "Oct 15 — Full paper draft" but Phase 4 runs Sep 29–Oct 26. That milestone sits at day 17 of 28, which is less than 60% through the phase. Either the milestone is aspirational and will slip, or the phase is mis-sized.

4. **Literature review is unowned.** It's mentioned as parallelisable but nobody is responsible for producing a literature review deliverable. This should be a concrete task with a deadline (e.g., literature review draft by Aug 15).

5. **No fallback scope.** What does the "minimum viable paper" look like if time runs short? Every phase should have a clear "stop loss" point — what you deliver if you hit the deadline with work remaining.

---

## 6. What I Would Change

### Priority 1: Start Literature Review Now (June 21–Jul 12)
This is free work that requires no data. Produce a synthesised literature review document (not just a reading list) by mid-July. This feeds directly into the Introduction and Discussion sections later.

### Priority 2: Storage & Compute Plan Before Data Arrives
Before July 13:
- Test the Pi's actual I/O throughput for typical data operations
- Set up a storage budget with hard limits
- Identify a cloud fallback (a \$50 Hetzner VM or Google Colab instance) if the Pi can't handle the data
- Pin library versions and containerise the environment

### Priority 3: Restructure Phases

| Phase | What | When | Duration |
|-------|------|------|----------|
| 0 | Literature review, compute setup, methodology framework | Jun 21–Jul 12 | 3 weeks |
| 1 | Data acquisition + cleaning + EDA | Jul 13–Aug 3 | 3 weeks |
| 2 | Core analysis (features, modelling, validation) | Aug 4–Sep 7 | 5 weeks |
| 3 | Final figures + robustness + reproducibility | Sep 8–Sep 28 | 3 weeks |
| *Buffer* | *Schedule insurance / catch-up* | *Sep 29–Oct 5* | *1 week* |
| 4 | Paper writing (all sections, no methods draft done early) | Oct 6–Oct 26 | 3 weeks |
| 5 | Presentation + submission prep | Oct 27–Nov 1 | 6 days |

The tradeoff: writing time drops from 4 weeks to 3 weeks. But this is offset by having the literature review done, the buffer capturing any Phase 1 delays, and Phase 3 being cleaner (no writing mixed in). Conor will need to write fast in October — 3 weeks for a full paper is possible with good preparation.

### Priority 4: Define "Plan B" Before Data Arrives
Before the data comes in, draft two story outlines:
- **Plan A (ambitious):** What the paper says if the data is rich and results are strong
- **Plan B (safe):** What the paper says if results are modest or null (descriptive statistics, case studies, framework contributions)

This prevents panic-pivoting in September.

### Priority 5: Build Reproducibility Into Every Phase
Use `renv`/`pip freeze` from Day 1. Track every data transformation in code. Set a hard requirement: at the end of every week, the entire pipeline up to that point must be reproducible from scratch on a fresh environment. This saves the "compiling supplementary materials" time sink at the end.

---

## Conclusion

The work plan is a solid first draft with the right instincts (parallelisation, dependency tracking, milestone thinking). But it overestimates what one person + one Raspberry Pi can achieve without explicit compute planning, underestimates the data exploration phase, and has no slack in the schedule. The structural tension between Phase 3 and Phase 4 needs resolution, and the literature review needs formalising.

With the adjustments above — starting the literature review immediately, planning compute/storage before data arrives, adding buffer, and defining a Plan B — the project is achievable. Without them, a Phase 1 delay could domino into a rushed paper and a weak presentation.

**Verdict:** Feasible with significant adjustments. High risk of timeline slippage as currently scoped.
