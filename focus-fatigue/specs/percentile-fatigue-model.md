# Percentile-Threshold Cognitive Fatigue Model

## Principle
No match halves. No minutes. No time-on-clock framing. Fatigue is defined solely by accumulated load percentiles. The question is: *at any moment in a match, does a player with high accumulated cognitive load show worse defensive performance than a player with low accumulated load?*

## How High/Low Groups Are Defined

1. For each (player, game, block), compute rolling cognitive load for each window type:
   - 10-min rolling: mean of preceding 2 blocks' pressure + reorientations + opponents_nearby
   - 15-min decay: exponentially weighted mean of all preceding blocks, tau=15
   - Half: mean of all preceding blocks in the current half
   - Full: mean of all preceding blocks from match start

2. For each window type, compute the 75th percentile of rolling_cog_load across ALL blocks.
   - Blocks above the 75th percentile → "high cognitive load"
   - Blocks below the 25th percentile → "low cognitive load"
   - Middle 50% discarded for clean contrast

3. No time variables. No phase. No minute markers.

## Models

**Model 1 — High vs Low Group Comparison (predictive)**
`defensive_quality ~ cog_load_group (high/low)`
- Do high-cognitive-load blocks show worse defensive quality?
- Run for each outcome × window type
- Report: mean + 95% CI for each group, difference + 95% CI

**Model 2 — With Physical Load Control**
`defensive_quality ~ cog_load_group + phys_load_group`
- Adding physical load group (also percentile-thresholded)
- Does the cognitive effect survive controlling for physical load?
- This is the dissociation test — are high cognitive load players worse BECAUSE of mental fatigue or just because they're running harder?

**Model 3 — Cognitive × Physical Interaction**
`defensive_quality ~ cog_load_group * phys_load_group`
- Is the cognitive effect different at different physical load levels?
- e.g., is cognitive fatigue worse when players are also physically exhausted?

## Reviewer Step

After the analysis runs, spawn a reviewer sub-agent to critique:
- Is the percentile threshold method valid? (biased? arbitrary?)
- Is the composite cognitive load measure a fair proxy for fatigue?
- Are we confounding demand with depletion?
- Are the controls sufficient?
- What alternative explanations are not ruled out?
- Is the statistical approach sound?

The reviewer produces `focus-fatigue/review/percentile-model-review.md`

## Output
- `focus-fatigue/outputs/analysis/percentile_fatigue_model.md`
- `focus-fatigue/outputs/analysis/percentile_fatigue_figure.png`
- `focus-fatigue/review/percentile-model-review.md` (reviewer audit)
- Push to origin/main
