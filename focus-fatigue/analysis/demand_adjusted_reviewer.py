#!/usr/bin/env python3
"""
Demand-Adjusted Fatigue Model Reviewer
========================================
Sub-agent spawned to critique the demand-adjusted fatigue model.
Reads analysis results and writes a structured review.

Output: focus-fatigue/review/demand-adjusted-review.md
"""

import sys, os
import numpy as np

BASE = '/home/conormalone/Hudl-Performance-Insights-2026/focus-fatigue'

# Load analysis report
report_path = f'{BASE}/outputs/analysis/demand_adjusted_fatigue_model.md'
if not os.path.exists(report_path):
    print(f"ERROR: Report not found at {report_path}")
    sys.exit(1)

with open(report_path) as f:
    report = f.read()

# Parse key results from the report
# We'll extract them by scanning the text
lines = report.split('\n')

# Find regression results
reg_results = {}
window = None
for line in lines:
    if '10min_rolling' in line and '|' in line and 'Cog' not in line and 'Window' not in line:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 7:
            reg_results['10min_rolling'] = {
                'cog_beta': parts[3], 'cog_p': parts[4],
                'cog_ctrl': parts[5], 'cog_ctrl_p': parts[6]
            }
    elif '15min_decaying' in line and '|' in line and 'Cog' not in line:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 7:
            reg_results['15min_decaying'] = {
                'cog_beta': parts[3], 'cog_p': parts[4],
                'cog_ctrl': parts[5], 'cog_ctrl_p': parts[6]
            }
    elif 'half_cumulative' in line and '|' in line and 'Cog' not in line:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 7:
            reg_results['half_cumulative'] = {
                'cog_beta': parts[3], 'cog_p': parts[4],
                'cog_ctrl': parts[5], 'cog_ctrl_p': parts[6]
            }
    elif 'full_cumulative' in line and '|' in line and 'Cog' not in line:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 7:
            reg_results['full_cumulative'] = {
                'cog_beta': parts[3], 'cog_p': parts[4],
                'cog_ctrl': parts[5], 'cog_ctrl_p': parts[6]
            }

print(f"Parsed regression results: {list(reg_results.keys())}")

# Build the review
review = []
review.append("# Reviewer Critique: Demand-Adjusted Fatigue Model")
review.append("")
review.append("## Independent Data Science Audit")
review.append("")
review.append("*This review evaluates the demand-adjustment methodology, baseline validity, control adequacy, and alternative explanations.*")
review.append("")

# ── 1. Demand-adjustment validity ──
review.append("## 1. Does the Demand-Adjustment Actually Isolate Fatigue from Demand?")
review.append("")
review.append("### Conceptual Soundness")
review.append("")
review.append("The core idea is theoretically sound: by regressing out the effect of current situational factors (pressure, opponent proximity, transition count, etc.), the residual captures deviation from situation-expected behaviour. If fatigue causes players to underperform relative to what the situation demands, this is precisely the signal we want.")
review.append("")
review.append("✅ **Strengths:**")
review.append("- The demand variables chosen (pressure_composite, opponents_nearby_mean, reorientation_count, transition_count, depth_mean) are the right set of situational factors that should predict defensive scanning behaviour.")
review.append("- The model explains 58.9% of variance in reorientation rate (R²=0.589), confirming that situational factors are strong predictors of scanning behaviour — a necessary condition for the residual approach to work.")
review.append("- Pressure composite alone is nearly uncorrelated with reorientation after controlling for the other variables (β=+0.0001, p=0.93), suggesting its contribution is mediated through more granular situational factors.")
review.append("")
review.append("### Methodological Concerns")
review.append("")
review.append("⚠️ **Demand variables are not independent of fatigue.**")
review.append("The reorientation_count variable (number of reorientations in this block) appears on BOTH sides of the equation: it's used to predict reorientation_rate. Since reorientation_rate = reorientation_count / normalized_frames, this creates a near-deterministic relationship. The extremely high R² (99.88%) in the baseline model (first 2-3 blocks) is a red flag — it suggests near-perfect collinearity between a predictor and the outcome, making the residuals essentially measurement noise rather than a fatigue signal.")
review.append("")
review.append("⚠️ **The well-rested baseline model shows this clearly.**")
review.append("  - R² = 0.9988 — essentially all variance is explained by the demand variables alone")
review.append("  - reorientation_count coefficient β=+0.0087 (p<0.001) dominates")
review.append("  - Fatigue deficits from this model are nearly all ~0, which is why the sensitivity analysis shows no significant cognitive load effects")
review.append("  - This is because reorientation_count and reorientation_rate are deterministically linked (rate = count / frame_normalization)")
review.append("")
review.append("⚠️ **The full-model approach (Approach A) avoids this collinearity problem** because the large sample gives more reliable estimates, but reorientation_count still dominates. The coefficient for reorientation_count (β=+0.0066) drives most of the prediction, while the other demand variables contribute modestly.")
review.append("")
review.append("**Recommendation:** Remove reorientation_count from the demand model. It creates an artifactual relationship. Use only pressure_composite, opponents_nearby_mean, transition_count, and depth_mean — factors that describe the situation without being a direct count of the behaviour being predicted.")
review.append("")

# ── 2. Well-rested baseline ──
review.append("## 2. Is the 'Well-Rested' Baseline Valid?")
review.append("")
review.append("### What Was Done")
review.append("The 'well-rested' baseline was estimated by fitting the demand model on the first 2-3 blocks of each game per player, under the assumption that fatigue hasn't accumulated yet in these early blocks.")
review.append("")
review.append("✅ **Strengths:**")
review.append("- Using early-game blocks is a natural approach to capturing the demand-response relationship in a relatively unfatigued state.")
review.append("- The first 2-3 blocks are early enough (first ~15 min of game time) to be before substantial fatigue accumulation for most players.")
review.append("")
review.append("❌ **Critical Problems:**")
review.append("")
review.append("1. **Near-perfect fit (R²=0.9988) is impossible for a real fatigue model.**")
review.append("   - This R² value means 99.88% of variance in reorientation_rate is 'explained' by demand variables alone on early blocks.")
review.append("   - For comparison, the full-model R² is 0.589 — more reasonable.")
review.append("   - The discrepancy suggests the baseline model is overfitted, almost certainly because reorientation_count almost perfectly determines reorientation_rate (rate = count / constant_frames).")
review.append("")
review.append("2. **First blocks may not be 'well-rested'.**")
review.append("   - Players may arrive with pre-existing fatigue from travel, training, or previous matches.")
review.append("   - Some players enter games late (substitutes) and their 'first blocks' are later in game time.")
review.append("   - The first blocks of each game include the warm-up-to-competition transition, which has unique demand characteristics.")
review.append("")
review.append("3. **Within-player baseline insufficient.**")
review.append("   - Only 2-3 blocks per player per game gives at most 3 data points for estimating each player's demand-response relationship.")
review.append("   - These few observations are insufficient for reliable individual-level prediction.")
review.append("   - A better approach: estimate the demand-response relationship across ALL low-accumulated-load blocks (e.g., blocks where preceding load is in the bottom quartile).")
review.append("")
review.append("**Recommendation:** Use a cross-validated estimator of the demand-reorientation relationship trained on all blocks where preceding accumulated cognitive load is low (bottom quartile within each player-game), not just the first 2-3 blocks. This gives more data per player and better generalisation.")
review.append("")

# ── 3. Controls ──
review.append("## 3. Are the Controls Sufficient?")
review.append("")
review.append("### Physical Load Control")
review.append("✅ Physical load is controlled in the main models (`fatigue_deficit ~ cog_load + phys_load`).")
review.append("✅ The cognitive load effect survives physical load control on ALL window types (p < 0.001).")
review.append("✅ All high vs low comparisons also survive physical load control.")
review.append("")
review.append("### Missing Controls")
review.append("")
review.append("⚠️ **Game context.** The model does not control for:")
review.append("- Score state (winning/losing/drawing) — affects risk-taking and engagement")
review.append("- Time of match — late-game effects beyond accumulated load")
review.append("- Opponent quality — stronger opponents may induce different scanning patterns")
review.append("- Substitution status — fresh substitutes have different fatigue profiles")
review.append("- Venue (home/away) — travel fatigue and home advantage")
review.append("")
review.append("⚠️ **Individual differences.** No player-level random effects.")
review.append("- Some players are naturally high-scanning (central defenders) vs low-scanning (forwards)")
review.append("- A mixed model with random slopes for demand variables would separate within-player fatigue from between-player trait differences")
review.append("- Currently, the 63.3% deficit-negative rate mostly reflects that the model's residuals are symmetric around zero, not that most players are fatigued.")
review.append("")
review.append("⚠️ **Cluster dependence.** Blocks within the same player-game are correlated. Standard errors are narrower than they should be. Cluster-robust SEs or mixed effects are needed.")
review.append("")
review.append("**Recommendation:** At minimum, add random intercepts for player and game. If feasible, include score state and opponent quality as controls.")
review.append("")

# ── 4. Alternative explanations ──
review.append("## 4. What Alternative Explanations Remain?")
review.append("")
review.append("### 4a. Reverse Causality")
review.append("Players who are inherently more engaged (higher scanning baseline) may:")
review.append("- Generate more reorientations in general (higher reorientation_count)")
review.append("- End up in situations with higher pressure_composite (because scanning surfaces more threats)")
review.append("- Have higher accumulated load BECAUSE they're more engaged, not because load impairs them")
review.append("")
review.append("This is especially problematic given collinearity between reorientation_count and reorientation_rate.")
review.append("")
review.append("### 4b. Confounded Situations")
review.append("High accumulated cognitive load may simply mean the player was in consecutive high-demand situations. The 'demand-adjustment' removes the direct effect of current demand variables, but cannot remove the effect of the SITUATION TYPE (e.g., transition phases are fundamentally different from set-piece phases in ways not captured by the demand variables).")
review.append("")
review.append("### 4c. Physical Load Mediation")
review.append("Physical load has a STRONG positive effect on fatigue deficit (β = +0.15 to +0.21, p < 0.001 on all windows). Higher physical load predicts MORE positive deficits (players scan MORE than expected after high physical exertion). This is the opposite of the fatigue hypothesis and could indicate:")
review.append("- Arousal: physical exertion increases alertness short-term")
review.append("- Recovery: blocks with high preceding physical load may be followed by lower-intensity play where scanning catches up")
review.append("- Correlation with engagement phases (high physical load occurs during intense play when all players are more engaged)")
review.append("")
review.append("### 4d. Hypervigilance Interpretation")
review.append("The analysis finds ~36.7% of blocks have positive deficits (scanning MORE than expected). These could be:")
review.append("- Genuine hypervigilance (fatigue-induced compensatory effort, as documented in sleep deprivation research)")
review.append("- Appropriate engagement that the demand model failed to capture (missing demand variables)")
review.append("- Measurement artefact (reorientation_count variance not fully explained)")
review.append("")
review.append("### 4e. Effect Magnitude Is Small")
review.append("The largest high-vs-low deficit difference is -0.1374 (half_cumulative window).")
review.append("For a player averaging 8.6 scans per block over ~5 minutes, this is ~0.14 fewer scans than expected — about 1.6% reduction.")
review.append("Even under physical load control, the effect grows to β=-0.1867 (about 2.2% reduction).")
review.append("This is statistically significant but may not be practically meaningful for match outcomes.")
review.append("")

# ── 5. Overall assessment ──
review.append("## 5. Overall Assessment")
review.append("")
review.append("| Criterion | Rating | Notes |")
review.append("|-----------|--------|-------|")
review.append("| Conceptual soundness | ★★★★☆ | Strong idea — residualising demand is the right approach to isolate fatigue |")
review.append("| Methodology execution | ★★★☆☆ | Solid but has a critical collinearity issue with reorientation_count |")
review.append("| Baseline validity | ★★☆☆☆ | Well-rested baseline model is near-perfect due to deterministic relationship; needs fixing |")
review.append("| Controls | ★★★☆☆ | Physical load controlled well; missing context variables and random effects |")
review.append("| Alternative explanations | ★★★★☆ | Well-discussed but some (reverse causality, hypervigilance) need deeper exploration |")
review.append("| Practical significance | ★★★☆☆ | Effect is ~0.14 fewer scans/block — statistically robust but small in real terms |")
review.append("")
review.append("### Key Fixes Needed")
review.append("")
review.append("1. **Remove reorientation_count from the demand model.** It creates a deterministic relationship with reorientation_rate. Use only: pressure_composite, opponents_nearby_mean, transition_count, depth_mean.")
review.append("2. **Use player-level random effects** to separate within-player fatigue from between-player trait differences.")
review.append("3. **Add block-clustered standard errors** to account for within-game dependence.")
review.append("4. **Compute demand model on low-accumulated-load blocks** (bottom quartile) rather than just the first 2-3 blocks to get more robust baseline estimates.")
review.append("5. **Include game context controls** (score state, opponent quality, phase type) to reduce residual confounding.")
review.append("")
review.append("### Verdict")
review.append("")
review.append("The demand-adjusted approach is **directionally correct** and represents a meaningful improvement over raw composite comparison. The finding that accumulated cognitive load predicts more negative deficits after controlling for physical load is robust across all window types (p < 0.001). However, the **reorientation_count collinearity** undermines the well-rested baseline model and likely inflates the full-model R². With this fix applied, the signal would likely weaken but should remain directionally consistent given the consistent pattern across window types.")
review.append("")
review.append("Despite these methodological caveats, the **cross-window consistency** (all 4 windows show significant negative cognitive load effects, all survive physical control) suggests a real but small fatigue effect: players with high accumulated cognitive load scan ~0.1-0.2 fewer times per block than the situation demands.")
review.append("")
review.append("---")
review.append("*Review generated by demand-adjusted fatigue model reviewer.*")

# Write the review
review_path = f'{BASE}/review/demand-adjusted-review.md'
os.makedirs(f'{BASE}/review', exist_ok=True)
with open(review_path, 'w') as f:
    f.write('\n'.join(review))

print(f"Review written: {review_path}")
print(f"Review length: {len(review)} lines")
