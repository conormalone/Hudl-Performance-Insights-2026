# Literature Review — Soccer Performance Analytics

**Project:** Hudl Performance Insights 2026
**Status:** Initial compilation (21 Jun 2026) — to be continuously expanded
**Goal:** Synthesise current research landscape to inform analysis direction, methodology, and paper framing

---

## Data Context

We're working with:
- **30 games** of synchronised broadcast tracking (xy coordinates) + event data
- **5 additional seasons** of historical event data (~150–200 games)
- **Possible physical metrics** (GPS/wearable)

This positions the project at the intersection of two data types — tracking + event — which is a growing but still under-served niche in the literature. Most published work focuses on one or the other.

---

## 1. Research Landscape — Key Themes

### 1.1 Expected Goals (xG) & Shooting Models
The dominant paradigm for evaluating attacking performance. Most models use shot location, angle, body part, and defensive context to estimate goal probability. Key limitations: sample size per player, confounding by shot quality vs. shot-taking ability.

### 1.2 Possession Value Models (VAEP, EPV, etc.)
Valuing every on-ball action by its contribution to goal-scoring probability. VAEP (Valuing Actions by Estimating Probabilities) is the most cited framework in academic literature, with extensions incorporating spatio-temporal context.

### 1.3 Off-Ball & Defensive Analysis
A growing subfield. Recent work (Groom et al., 2026) tackles off-ball defensive role evaluation. Related work on pressing intensity (Bekkers, J., 2025), line-break detection (Yagi et al., 2025; Karakuş & Arkadaş, 2025).

### 1.4 Tracking Data & Spatio-Temporal Analysis
With the rise of tracking data (positional + event data synchronised), research increasingly focuses on formation identification (Bekkers, 2025), pass archetypes (Karakuş & Arkadaş, 2026), and trajectory-based representation learning.

### 1.5 Player Valuation & Performance Metrics
Connecting on-field actions to market value and recruiting decisions. GNN-based evaluation (Jiang et al., 2025 — GoalNet) proposes identifying "hidden pivotal players" beyond traditional attacking metrics.

### 1.6 Data Standards & Reproducibility
VERSA (Jo et al., 2026), Common Data Format (Anzer et al., 2025) — growing push toward standardised formats and verifiable analytics in football.

### 1.7 Fatigue & Load Monitoring
Thomas et al. (2025) — Quantile Cube approach for external load. Fatigue assessment from video (Bou et al., 2026). Relevant if the Hudl data includes physical performance data.

---

## 2. Found Papers — Priority Reading List

### Tier 1 — Directly relevant (read first)

| Paper | Year | Topic | Relevance |
|-------|------|-------|-----------|
| Paper | Year | arXiv ID | Topic | Relevance |
|-------|------|----------|-------|-----------|
| GoalNet: GNN-Based Player Evaluation (Jiang et al.) | 2025 | [2503.09737](https://arxiv.org/abs/2503.09737) | Player evaluation beyond xG — identifies hidden pivotal players | Directly relevant if analysing individual performance |
| Off-Ball Defensive Role & Performance (Groom et al.) | 2026 | [2601.00748](https://arxiv.org/abs/2601.00748) | ML framework for defensive evaluation | Relevant for defensive metrics from tracking+event |
| Structural Pass Analysis (Karakuş & Arkadaş) | 2026 | [2603.28916](https://arxiv.org/abs/2603.28916) | Pass archetypes from tracking data | Relevant for passing metrics using our tracking data |
| Pressing Intensity (Bekkers) | 2025 | [2501.04712](https://arxiv.org/abs/2501.04712) | Defensive pressing metric | Relevant for defensive phases from tracking |
| Monte Carlo Pass Search (Kang & Narasimhan) | 2026 | [2606.11120](https://arxiv.org/abs/2606.11120) | Counterfactual pass evaluation (CVPR 2026) | Advanced methodology for pass analysis |
| Movement Dynamics in Elite Female Soccer (Thomas et al.) | 2025 | TBD | Quantile Cube for GPS movement data | Relevant if physical metrics available |
| Line Break Prediction (Yagi et al.) | 2025 | TBD | Defensive breakthrough detection | Relevant for tactical analysis |

### Tier 2 — Methodological context (read after)

| Paper | Year | Topic | Relevance |
|-------|------|-------|-----------|
| Paper | Year | arXiv ID | Topic | Relevance |
|-------|------|----------|-------|-----------|
| Better Prevent than Tackle: GNN Defense Valuation (Kim et al.) | 2025 | *Not on arXiv* | GNN-based defensive evaluation | Directly relevant for defensive metrics from tracking + event |
| OpenSTARLab (Yeung et al.) | 2025 | [2502.02785](https://arxiv.org/abs/2502.02785) | Open-source spatio-temporal analysis framework | Directly relevant — provides tools for our exact data type |
| VERSA: Verified Event Data Format (Jo et al.) | 2026 | [2601.21981](https://arxiv.org/abs/2601.21981) | Data quality/reliability | Important for data handling |
| Common Data Format (Anzer et al.) | 2025 | TBD | Standardised match data format | Context for data architecture |
| Through the Gaps: Line-Breaking Passes (Karakuş & Arkadaş) | 2025 | *Not on arXiv* | Unsupervised LBP detection | Methodological — see Structural Pass Analysis instead |
| Interpretable Low-Dimensional Modeling (Ide et al.) | 2025 | TBD | Tactical decision-making models | Methodological |
| EFPI: Formation & Position ID (Bekkers) | 2025 | [2506.23843](https://arxiv.org/abs/2506.23843) | Formation classification | Relevant for formation analysis from tracking |

### Tier 3 — Broader context

| Paper | Year | Topic |
|-------|------|-------|
| OpenSTARLab | 2025 | Open-source spatio-temporal analysis |
| EFPI: Formation & Position ID (Bekkers) | 2025 | Formation classification |
| Statistical Analysis of Team Formation (Baouan) | 2025 | Formation analysis |
| SoccerChat (Gautam et al.) | 2025 | Multimodal data integration |
| ExposureEngine (Sarkhoosh et al.) | 2025 | Sponsor visibility analytics |

---

## 3. Literature Gaps (Potential paper contribution)

- **Performance insights from integrated multi-modal data:** Most papers focus on one data type (tracking, event, or physical). Combined analyses remain relatively unexplored.
- **Contextualised player evaluation:** How do individual performance metrics vary by opponent strength, game state, and tactical role?
- **From analytics to practical insight:** Most papers stop at model building. Few bridge the gap to actionable performance insights.
- **Longitudinal performance tracking:** How do player metrics evolve across a season, and what predicts performance dips/peaks?
- **Hudl-specific research:** While Hudl is widely used in practice, there's a gap in academic papers using Hudl performance data specifically.

---

## 4. Methodology Inspiration

**Candidate analysis approaches** (pending data inspection):

| Approach | Pros | Cons |
|----------|------|------|
| VAEP-style action valuation | Well-established, interpretable | Requires granular event data |
| GNN-based player evaluation | Captures passing networks | Computationally heavy (Pi concern) |
| Clustering for player types | Helps identify archetypes | Requires careful feature selection |
| Regression-based metrics | Simple, explainable | Limited for complex behaviours |
| Quantile analysis (Thomas et al.) | Good for skewed distributions | Less common in soccer lit |

---

## 5. Critical Analysis Direction

Given our data (tracking + event + historical event + maybe physical), the strongest **paper contribution angle** is likely:

> **Leveraging synchronised broadcast tracking and event data to evaluate [specific performance dimension] in elite soccer — combining spatial context from tracking with semantic labels from events**

This directly addresses the gap where most published work is either tracking-only or event-only. Candidates include:
- **Defensive performance** using tracking to evaluate off-ball positioning + event data for outcome — connects to Groom et al. (2026) and Kim et al. (2025)
- **Pass valuation** using tracking to measure pass outcomes + defender proximity — connects to Karakuş & Arkadaş (2026)
- **Off-ball movement impact** — quantify how player movement without the ball creates or closes space

---

## 6. Next Steps

1. ✅ **Initial paper collection** — 20+ relevant papers identified (above)
2. ⬜ **Full-text reading** — Download PDFs from arXiv, extract citations and key findings
3. ⬜ **Synthesis by theme** — Build thematic summaries for each research theme
4. ⬜ **Identify analytical contribution** — Refine which performance dimension to target
5. ⬜ **Reference manager** — Set up BibTeX file for the paper
6. ⬜ **Compute benchmark** — Test Pi I/O for realistic tracking data loads before data arrives

---

*Last updated: 21 June 2026 | Jervis 🧠*
