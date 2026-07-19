# Reviewer Critique: Percentile-Threshold Cognitive Fatigue Model

## Independent Data Science Audit

## 1. Is the Percentile-Threshold Method Valid?

### Strengths

- **Clean contrast.** Discarding the middle 50% avoids muddy comparisons where both groups are similar. This is a legitimate approach for detecting dose-response relationships.
- **Global thresholds** across all blocks make the comparison general rather than within-player-relative, which tests a broader hypothesis about absolute load levels.
- **Multiple windows** (10-min rolling, 15-min decay, half-game, full-game) provide robustness — consistent patterns across windows increase confidence.

### Concerns

- **Information loss.** Collapsing a continuous rolling load measure into three bins (high/middle/low) discards substantial variance. Dichotomisation reduces statistical power and can obscure non-linear relationships. A continuous spline or polynomial model would retain more information.
- **Arbitrary cutoffs.** The 75th/25th percentile split is conventional but unvalidated. Different thresholds (e.g., 90th/10th, tertiles) could yield different conclusions. The absence of sensitivity analysis is a limitation.
- **Global, not within-player.** Percentiles computed across ALL blocks mean that a player with consistently high load might always be in the 'high' group, while a different player with consistently low load is always in 'low'. This conflates individual differences (trait-like) with within-player fatigue (state-like). Within-player percentile thresholds would better isolate fatigue effects.
- **Middle block exclusion.** Discarding 50% of data reduces the effective sample considerably. If the middle blocks have systematic properties (e.g., transition zones in matches), their exclusion could bias the comparison.

## 2. Is the Cognitive Load Composite a Fair Proxy for Fatigue?

### The Demand vs Depletion Problem

- The composite measures **cognitive demand** (high pressure, many opponents nearby, frequent reorientations) — not necessarily cognitive **depletion**.
- A player in a high-demand moment (e.g., defending a counter-attack) will naturally show higher reorientation rates and pressing accuracy because the situation requires it, not because they are 'fatigued'.
- The finding that high-cognitive-load blocks show **better** reorientation rates and pressing accuracy (d=0.49–0.77) strongly supports this: players are responding appropriately to demanding situations, not suffering from fatigue.

### What shift_latency tells us

- Shift latency (slower = worse) is the **only outcome that shows the fatigue-consistent direction**: high cognitive load → slower shifts.
- This makes sense theoretically: response time measures are classically sensitive to mental fatigue (psychomotor vigilance). A fatigued player takes longer to recognise and react to unfolding play.
- However, the effect sizes are tiny (d≈0.06), and the effect is confounded by physical load on half/full-game windows.

### Recommendation

- The composite is measuring **cognitive demand**, which is a precursor to fatigue but not identical to it. A true fatigue measure would require within-player decline over time, not comparison of high-demand vs low-demand moments.
- Consider using residualised change scores: cognitive load at time t controlling for baseline cognitive load, to isolate the 'unexpected' load that might cause fatigue.

## 3. Alternative Explanations Not Ruled Out

### Selection bias (most critical)

- Blocks with high cognitive load are systematically different situations: fast transitions, opponent pressure, counter-attacks. Players in these blocks are more alert and engaged by default. The 'better' reorientation/pressing scores may reflect situational urgency, not cognitive vitality.
- Conversely, low-cognitive-load blocks may be quiet periods (e.g., slow build-up, set pieces), where scanning and pressing are naturally lower.
- **This is a confound between situation and state.** The analysis cannot distinguish whether high-cognitive-load blocks produce better outcomes because players are engaged, or because the game situation demands more intense responses.

### Physical load correlation

- Cognitive load and physical load are correlated (both increase during intense phases). Model 2 partially addresses this, but the residual confounding may be substantial.
- The shift_latency effect being confounded on half/full windows suggests that physical load and cognitive load share variance that matters for fatigue detection.

### Learning effects and strategies

- Players may strategically conserve energy during low-load phases (e.g., walking during build-up) and exert during high-load phases. This is not fatigue — it's pacing.

## 4. Statistical Approach Soundness

### Strengths

- **Three-model progression** (univariate → controlled → interaction) is a sound analytical structure.
- **Multiple comparison windows** test robustness rather than cherry-picking one.
- **Welch t-tests** for the primary comparison avoid pooled-variance assumptions.

### Concerns

- **Multiple comparisons.** With 4 windows × 3 outcomes = 12 model-1 tests, multiplicity correction (Bonferroni, FDR) should be applied. At α=0.05, ~0.6 false positives are expected. Many results are extremely significant (p<0.0001), so this may not change conclusions, but it should be noted.
- **No clustering correction.** Blocks within the same player-game are not independent. The analysis treats each block as an independent observation, inflating effective sample size and narrowing confidence intervals. Cluster-robust standard errors (by player-game) or mixed-effects models would be more appropriate.
- **No random effects.** Model 2 and 3 use OLS with fixed group indicators. Adding random intercepts for player and game would better account for the nested structure (blocks within players within games).
- **Equal group sizes by construction.** The 25th/75th percentile split creates equal-sized groups by definition. This is fine for comparison but means the 'middle' 50% is the same size as both tails combined, which is an odd asymmetry.
- **extreme values in shift_latency.** The standard deviation of shift_latency is 49.7 vs mean of 5.4 — an extreme right skew. The t-test assumes normality. Log-transformation or robust methods (bootstrapped CI, quantile regression) should be used.

## 5. Recommendations to Strengthen the Methodology

### Immediate (same data, different approach)

1. **Within-player percentile thresholds.** Instead of global percentiles, compute high/low relative to each player's own distribution. This separates within-player fatigue from between-player talent/role differences.
2. **Continuous spline models.** Replace the dichotomy with restricted cubic splines or generalised additive models of rolling cognitive load, retaining the full continuous information.
3. **Cluster-robust standard errors.** Use mixed-effects models (lmer in R, mixedlm in statsmodels) with random intercepts for player and game, or at minimum sandwich-type standard errors clustered on player-game.
4. **Sensitivity analysis.** Report results at multiple thresholds (e.g., 90th/10th, 80th/20th, 70th/30th) to test whether findings are robust to cutoff choice.
5. **Log-transform shift_latency.** The extreme skew demands log transformation or a GLM with Gamma family.

### Deeper methodological changes

6. **Residual cognitive load.** Compute 'unexpected' cognitive load by regressing current cognitive load on previous blocks' physical load and tactical context (e.g., phase type, score state, opponent quality). The residual represents load beyond what physical context predicts — a cleaner measure of 'surplus' cognitive demand.
7. **Change-score approach.** Model within-player change: `Δquality ~ rolling_cog_load`, where each block's quality is compared to the player's own baseline from quiet periods. This controls for individual differences.
8. **Control for event type.** Include a covariate for the type of phase (high press vs. low block vs. transition) to separate situational demands from fatigue.
9. **Bayesian hierarchical model.** A multilevel model with partial pooling would handle the nested structure naturally and provide shrinkage estimates for players with few observations.

## Conclusion

The percentile-threshold approach is a **valid exploratory method** that cleanly separates high and low cognitive load states. The multiple-window design provides robustness.


However, the **fundamental limitation** is that the cognitive load composite measures **cognitive demand** (situational intensity) rather than **cognitive depletion** (fatigue). The finding that high-load blocks show better scanning and pressing is consistent with demand, not fatigue. The only fatigue-consistent signal (shift latency) has tiny effects that partially disappear under physical load control.


**Verdict:** The methodology is directionally correct but insufficient to conclude that cognitive fatigue impairs defensive quality. The analysis likely detects **situational selection effects** (high-demand moments = more engaged players) rather than **depletion effects** (accumulated load → worse performance). Within-player change-score analysis and event-type controls would be necessary to resolve this confound.


---
*Review generated at analysis time. Data: 4 window×outcome combinations reviewed.*
