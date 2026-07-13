# arXiv PDF Download Report

**Date:** 2026-06-21  
**Task:** Literature review assets for Hudl Performance Insights 2026 (soccer analytics research project)

---

## Summary

| Status | Count |
|--------|-------|
| Downloaded (confirmed list) | 5 |
| Downloaded (via search) | 5 |
| Not found on arXiv | 2 |
| **Total downloaded** | **10** |

---

## Batch 1: Confirmed Papers (from task spec)

All 5 downloaded successfully via `curl -L -o` from `https://arxiv.org/pdf/`.

| # | arXiv ID | File | Title | Size |
|---|----------|------|-------|------|
| 1 | 2601.00748 | `2601.00748.pdf` | A Machine Learning Framework for Off Ball Defensive Role and Performance Evaluation in Football (Groom et al.) | 3.8 MB |
| 2 | 2503.09737 | `2503.09737.pdf` | Unveiling Hidden Pivotal Players with GoalNet: A GNN-Based Soccer Player Evaluation System (Jiang et al.) | 686 KB |
| 3 | 2501.04712 | `2501.04712.pdf` | Pressing Intensity: An Intuitive Measure for Pressing in Soccer (Bekkers) | 2.5 MB |
| 4 | 2603.28916 | `2603.28916.pdf` | Structural Pass Analysis in Football: Learning Pass Archetypes and Tactical Impact from Spatio-Temporal Tracking Data (Karakuş & Arkadaş) | 5.5 MB |
| 5 | 2606.11120 | `2606.11120.pdf` | Monte Carlo Pass Search: Using Trajectory Generation for 3D Counterfactual Pass Evaluation in Football (Kang & Narasimhan) | 1.9 MB |

## Batch 2: Papers Found via arXiv Search

These 5 papers were identified by searching `https://arxiv.org/search/` for the requested paper titles, then downloading.

| # | arXiv ID | File | Title | Size |
|---|----------|------|-------|------|
| 6 | 2502.02785 | `2502.02785.pdf` | OpenSTARLab: Open Approach for Spatio-Temporal Agent Data Analysis in Soccer (Yeung et al., Feb 2025) | 4.3 MB |
| 7 | 2503.11815 | `2503.11815.pdf` | Movement Dynamics in Elite Female Soccer Athletes: The Quantile Cube Approach (Thomas & Hannig, Mar 2025) | 4.6 MB |
| 8 | 2601.21981 | `2601.21981.pdf` | VERSA: Verified Event Data Format for Reliable Soccer Analytics (Jo et al., Jan 2026) | 917 KB |
| 9 | 2506.23843 | `2506.23843.pdf` | EFPI: Elastic Formation and Position Identification in Football (Soccer) using Template Matching and Linear Assignment (Bekkers, Jun 2025) | 1.1 MB |
| 10 | 2511.00121 | `2511.00121.pdf` | Analysis of Line Break prediction models for detecting defensive breakthrough in football (Yagi et al., Oct 2025) | 1.2 MB |

## Papers Not Found on arXiv

| Paper Title | Authors (est.) | Notes |
|-------------|----------------|-------|
| Through the Gaps | Karakuş & Arkadaş (Jun 2025) | No results on arXiv with this exact title. May be a different publication venue or unpublished. The authors' "Structural Pass Analysis" (2603.28916) was found instead. |
| Better Prevent than Tackle | Kim et al. (Dec 2025) | No results on arXiv with this exact title. Could be at a different venue or unpublished preprint. |

---

## Text Extraction Notes

**PDF text extraction was not possible** on this system — `pdftotext`/`poppler-utils`, `pypdf`, `PyPDF2`, `pdfminer`, `pdfplumber`, and `fitz`/`PyMuPDF` are all unavailable, and `sudo` is not accessible to install packages.

**Workaround used:** Abstracts were retrieved via the **arXiv OAI-PMH API** (`https://export.arxiv.org/api/query`) for all 10 papers. This provides clean plain-text abstracts and metadata without needing PDF parsing.

### Abstract Excerpts (via arXiv API)

1. **2601.00748** (Groom et al.) — Introduces a covariate-dependent Hidden Markov Model (CDHMM) for off-ball defensive role assignment and performance evaluation on corner kicks. Uses label-free man-marking/zonal assignment and role-conditioned ghosting for counterfactual defensive credit attribution.

2. **2503.09737** (Jiang et al.) — GoalNet: a GNN-based framework that assigns individual credit for changes in expected threat (xT), capturing overlooked contributions from defensive/transitional plays. Uses Graph Attention Networks and Transformer-based models.

3. **2501.04712** (Bekkers) — Pressing Intensity: quantifies pressing in soccer using positional tracking data, player velocities, movement directions, and reaction times. Extends Spearman's Pitch Control model with time-to-intercept measures and logistic function transformation.

4. **2603.28916** (Karakuş & Arkadaş) — Structural Pass Analysis: three complementary metrics (Line Bypass Score, Space Gain Metric, Structural Disruption Index) combined into Tactical Impact Value (TIV). Clustering reveals four pass archetypes. Data from 2022 FIFA World Cup.

5. **2606.11120** (Kang & Narasimhan) — Monte Carlo Pass Search (MCPS): recasts pass evaluation as MCTS-like evaluation using a world model, value model, and policy over counterfactual actions. Uses Bundesliga 3D tracking data and SMART trajectory generator.

6. **2502.02785** (Yeung et al.) — OpenSTARLab: open-source framework for spatio-temporal agent data analysis. Includes Pre-processing Package (standardized event/tracking data formats), Event Modeling Package (deep learning event prediction), and RLearn Package (reinforcement learning).

7. **2503.11815** (Thomas & Hannig) — Quantile Cube: 3D summary representation for external load analysis using GPS movement data from elite female soccer athletes across 23 matches. Uses Dirichlet-multinomial regression for movement profile analysis.

8. **2601.21981** (Jo et al.) — VERSA: state-transition model for verifying event stream data integrity in soccer. Found 18.81% logical inconsistencies in K League 1 data. Demonstrated improvements in cross-provider consistency and downstream VAEP performance.

9. **2506.23843** (Bekkers) — EFPI: formation recognition and player position assignment using template matching and linear sum assignment. Handles individual frames and game segments. Open-source via unravelsports Python package.

10. **2511.00121** (Yagi et al.) — Line Break prediction using XGBoost with 189 features (player positions, velocities, spatial configurations). AUC of 0.982 on J1 League 2023 data. SHAP analysis highlights offensive player speed and defensive gaps.

---

## File Listing

```
/mnt/project_data/project/assets/
  2501.04712.pdf   (2.5 MB)  Pressing Intensity
  2502.02785.pdf   (4.3 MB)  OpenSTARLab
  2503.09737.pdf   (686 KB)  GoalNet
  2503.11815.pdf   (4.6 MB)  Quantile Cube
  2506.23843.pdf   (1.1 MB)  EFPI
  2511.00121.pdf   (1.2 MB)  Line Break Prediction
  2601.00748.pdf   (3.8 MB)  Off-Ball Defensive Role
  2601.21981.pdf   (917 KB)  VERSA
  2603.28916.pdf   (5.5 MB)  Structural Pass Analysis
  2606.11120.pdf   (1.9 MB)  Monte Carlo Pass Search
```

**Total: 10 PDFs | 26.4 MB**
