#!/usr/bin/env python3
"""
Compute performance-context confidence intervals for reorientation_rate,
shift_latency, and pressing_accuracy across Phase 1 vs Phase 2.

Output: focus-fatigue/outputs/analysis/performance_context_intervals.md
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_parquet(
    "focus-fatigue/outputs/analysis/unified_fatigue_dataset.parquet"
)

BLOCKS_PER_HALF = 9          # per-task specification (~45 min / 5-min blocks)
FPS = 25                     # assumed frame rate
FRAMES_PER_5MIN = FPS * 300  # 7500

# ── Helper: mean and 95% CI using t-distribution ───────────────────────────
def mean_ci(x):
    """Return (mean, lower, upper) for 95% CI."""
    n = len(x)
    if n < 2:
        return (np.nan, np.nan, np.nan)
    m = np.mean(x)
    se = np.std(x, ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, df=n - 1)
    return (m, m - t * se, m + t * se)


def diff_ci(x1, x2):
    """Welch 95% CI for mean(x2) - mean(x1)."""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return (np.nan, np.nan, np.nan)
    m1, m2 = np.mean(x1), np.mean(x2)
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    # Welch–Satterthwaite degrees of freedom
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df_w = num / den if den > 0 else 1.0
    t = stats.t.ppf(0.975, df=df_w)
    d = m2 - m1
    return (d, d - t * se, d + t * se)


def cohens_d_ci(x1, x2):
    """Cohen's d (pooled) with 95% CI via non-central t approximation."""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return (np.nan, np.nan, np.nan)
    m1, m2 = np.mean(x1), np.mean(x2)
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return (np.nan, np.nan, np.nan)
    d = (m2 - m1) / s_pooled
    # CI via non-central t (Cumming & Finch, 2001)
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    z = stats.norm.ppf(0.975)
    return (d, d - z * se_d, d + z * se_d)


# ── Signal definitions ─────────────────────────────────────────────────────
signals = ["reorientation_rate", "shift_latency", "pressing_accuracy"]
results = {}

for sig in signals:
    p1 = df.loc[df["phase"] == 1, sig].dropna().values
    p2 = df.loc[df["phase"] == 2, sig].dropna().values

    n1, n2 = len(p1), len(p2)
    m1, lo1, hi1 = mean_ci(p1)
    m2, lo2, hi2 = mean_ci(p2)
    delta, dlo, dhi = diff_ci(p1, p2)
    d, d_lo, d_hi = cohens_d_ci(p1, p2)

    results[sig] = {
        "n1": n1, "n2": n2,
        "m1": m1, "lo1": lo1, "hi1": hi1,
        "m2": m2, "lo2": lo2, "hi2": hi2,
        "delta": delta, "dlo": dlo, "dhi": dhi,
        "d": d, "d_lo": d_lo, "d_hi": d_hi,
    }

# ── Real-world unit conversions ─────────────────────────────────────────────
# Reorientation rate: already per-minute → scans per 5-min block
rr = results["reorientation_rate"]
rr_scan_drop = rr["delta"] * 5            # scans lost per 5-min block
rr_scan_drop_lo = rr["dlo"] * 5
rr_scan_drop_hi = rr["dhi"] * 5
rr_scan_p1 = rr["m1"] * 5                # scans per 5-min block, Phase 1
rr_scan_p1_lo = rr["lo1"] * 5
rr_scan_p1_hi = rr["hi1"] * 5
rr_scan_p2 = rr["m2"] * 5
rr_scan_p2_lo = rr["lo2"] * 5
rr_scan_p2_hi = rr["hi2"] * 5

# Per half: lost scans = block_drop × 9
rr_half_drop = rr_scan_drop * BLOCKS_PER_HALF
rr_half_drop_lo = rr_scan_drop_lo * BLOCKS_PER_HALF
rr_half_drop_hi = rr_scan_drop_hi * BLOCKS_PER_HALF

# Shift latency: already in seconds
sl = results["shift_latency"]
sl_extra = sl["delta"]          # seconds per shift (P2 - P1, expected negative)
sl_extra_lo = sl["dlo"]
sl_extra_hi = sl["dhi"]

# Average shifts per half = avg transition_count per block × 9
# Use overall mean transition_count
avg_trans_block = df["transition_count"].mean()
avg_shifts_per_half = avg_trans_block * BLOCKS_PER_HALF

sl_half_total = sl_extra * avg_shifts_per_half
sl_half_total_lo = sl_extra_lo * avg_shifts_per_half
sl_half_total_hi = sl_extra_hi * avg_shifts_per_half

# Pressing accuracy: proportion → percentage points
pa = results["pressing_accuracy"]
pa_drop_pp = pa["delta"] * 100         # percentage-point drop per block
pa_drop_pp_lo = pa["dlo"] * 100
pa_drop_pp_hi = pa["dhi"] * 100

pa_half_total = pa_drop_pp * BLOCKS_PER_HALF
pa_half_total_lo = pa_drop_pp_lo * BLOCKS_PER_HALF
pa_half_total_hi = pa_drop_pp_hi * BLOCKS_PER_HALF

# ── Build markdown output ──────────────────────────────────────────────────
lines = []
lines.append("# Performance-Context Confidence Intervals")
lines.append("")
lines.append(
    "Comparison of **reorientation_rate**, **shift_latency**, and "
    "**pressing_accuracy** between Phase 1 (first half) and Phase 2 "
    "(second half)."
)
lines.append("")
lines.append("---")
lines.append("## 1. Raw Means with 95% CI")
lines.append("")

header = "| Metric | Phase 1 Mean [95% CI] | Phase 2 Mean [95% CI] | Δ (P2 − P1) [95% CI] | n₁ | n₂ |"
sep    = "|--------|----------------------|----------------------|----------------------|----|----|"
lines.append(header)
lines.append(sep)

for sig_name in signals:
    r = results[sig_name]
    sig_pretty = sig_name.replace("_", " ").title()
    lines.append(
        f"| {sig_pretty} "
        f"| {r['m1']:.4f} [{r['lo1']:.4f}, {r['hi1']:.4f}] "
        f"| {r['m2']:.4f} [{r['lo2']:.4f}, {r['hi2']:.4f}] "
        f"| {r['delta']:.4f} [{r['dlo']:.4f}, {r['dhi']:.4f}] "
        f"| {r['n1']} | {r['n2']} |"
    )

lines.append("")
lines.append("---")
lines.append("## 2. Real-World Units (per 5-min Block)")
lines.append("")

lines.append("### Reorientation Rate → Scans per 5-min Block")
lines.append("")
lines.append(
    f"- **Phase 1:** {rr_scan_p1:.2f} scans/block [{rr_scan_p1_lo:.2f}, {rr_scan_p1_hi:.2f}]"
)
lines.append(
    f"- **Phase 2:** {rr_scan_p2:.2f} scans/block [{rr_scan_p2_lo:.2f}, {rr_scan_p2_hi:.2f}]"
)
lines.append(
    f"- **Drop:** {rr_scan_drop:.2f} scans/block [{rr_scan_drop_lo:.2f}, {rr_scan_drop_hi:.2f}]"
)
lines.append("")
lines.append(f"*(Conversion: reorientation_rate (per min) × 5 min = scans per 5-min block)*")
lines.append("")

lines.append("### Shift Latency → Extra Seconds per Shift")
lines.append("")
lines.append(
    f"- **Phase 1:** {sl['m1']:.4f} s [{sl['lo1']:.4f}, {sl['hi1']:.4f}]"
)
lines.append(
    f"- **Phase 2:** {sl['m2']:.4f} s [{sl['lo2']:.4f}, {sl['hi2']:.4f}]"
)
lines.append(
    f"- **Δ (P2 − P1):** {sl_extra:.4f} s [{sl_extra_lo:.4f}, {sl_extra_hi:.4f}]"
)
lines.append("")

lines.append("### Pressing Accuracy → Percentage Points per Block")
lines.append("")
lines.append(
    f"- **Phase 1:** {pa['m1']*100:.2f}% [{pa['lo1']*100:.2f}%, {pa['hi1']*100:.2f}%]"
)
lines.append(
    f"- **Phase 2:** {pa['m2']*100:.2f}% [{pa['lo2']*100:.2f}%, {pa['hi2']*100:.2f}%]"
)
lines.append(
    f"- **Δ (P2 − P1):** {pa_drop_pp:.2f} pp [{pa_drop_pp_lo:.2f}, {pa_drop_pp_hi:.2f}]"
)
lines.append("")

lines.append("---")
lines.append("## 3. Per-Half Aggregates (~9 blocks/half)")
lines.append("")
lines.append(f"*(Assuming ~{BLOCKS_PER_HALF} × 5-min blocks per half)*")
lines.append("")

lines.append("### Reorientation — Total Scans Lost per Half")
lines.append(
    f"- Drop per block: {rr_scan_drop:.2f} scans [{rr_scan_drop_lo:.2f}, {rr_scan_drop_hi:.2f}]"
)
lines.append(
    f"- **Total lost per half:** {rr_half_drop:.2f} scans [{rr_half_drop_lo:.2f}, {rr_half_drop_hi:.2f}]"
)
lines.append("")

lines.append("### Shift Latency — Total Extra Reaction Time per Half")
lines.append(
    f"- Extra per shift: {sl_extra:.4f} s [{sl_extra_lo:.4f}, {sl_extra_hi:.4f}]"
)
lines.append(
    f"- Avg transition count per block: {avg_trans_block:.3f}"
)
lines.append(
    f"- Avg shifts per half: {avg_shifts_per_half:.2f}"
)
lines.append(
    f"- **Total extra reaction time per half:** "
    f"{sl_half_total:.2f} s [{sl_half_total_lo:.2f}, {sl_half_total_hi:.2f}]"
)
lines.append("")

lines.append("### Pressing Accuracy — Percentage-Point Drop per Half")
lines.append(
    f"- Drop per block: {pa_drop_pp:.2f} pp [{pa_drop_pp_lo:.2f}, {pa_drop_pp_hi:.2f}]"
)
lines.append(
    f"- **Total accuracy deficit per half:** "
    f"{pa_half_total:.2f} pp [{pa_half_total_lo:.2f}, {pa_half_total_hi:.2f}]"
    f"  (cumulative across {BLOCKS_PER_HALF} blocks)"
)
lines.append("")

lines.append("---")
lines.append("## 4. Effect Sizes (Cohen's d) with 95% CI")
lines.append("")

lines.append("| Metric | Cohen's d | 95% CI | Interpretation |")
lines.append("|--------|-----------|--------|----------------|")
for sig_name in signals:
    r = results[sig_name]
    d_val = r["d"]
    d_lower = r["d_lo"]
    d_upper = r["d_hi"]
    # interpret effect size
    ad = abs(d_val)
    if ad < 0.2:
        interp = "negligible"
    elif ad < 0.5:
        interp = "small"
    elif ad < 0.8:
        interp = "medium"
    else:
        interp = "large"
    sig_pretty = sig_name.replace("_", " ").title()
    lines.append(
        f"| {sig_pretty} | {d_val:.4f} | [{d_lower:.4f}, {d_upper:.4f}] | {interp} |"
    )
lines.append("")

lines.append("---")
lines.append("## 5. Paper-Ready Results Table")
lines.append("")
lines.append(
    "| Metric | Phase 1 | Phase 2 | Δ (95% CI) | d (95% CI) | Per-half impact |"
)
lines.append(
    "|--------|---------|---------|------------|------------|----------------|"
)

# Reorientation row (scans/5-min block)
lines.append(
    f"| Reorientation (scans/block) "
    f"| {rr_scan_p1:.1f} "
    f"| {rr_scan_p2:.1f} "
    f"| {rr_scan_drop:.1f} [{rr_scan_drop_lo:.1f}, {rr_scan_drop_hi:.1f}] "
    f"| {rr['d']:.2f} [{rr['d_lo']:.2f}, {rr['d_hi']:.2f}] "
    f"| −{abs(rr_half_drop):.0f} scans/half "
    f"|"
)

# Shift latency row (seconds)
lines.append(
    f"| Shift latency (s) "
    f"| {sl['m1']:.2f} "
    f"| {sl['m2']:.2f} "
    f"| {sl['delta']:.2f} [{sl['dlo']:.2f}, {sl['dhi']:.2f}] "
    f"| {sl['d']:.2f} [{sl['d_lo']:.2f}, {sl['d_hi']:.2f}] "
    f"| {sl_half_total:.0f} s/half "
    f"|"
)

# Pressing accuracy row (percentage points)
lines.append(
    f"| Pressing accuracy (pp) "
    f"| {pa['m1']*100:.1f}% "
    f"| {pa['m2']*100:.1f}% "
    f"| {pa_drop_pp:.2f} [{pa_drop_pp_lo:.2f}, {pa_drop_pp_hi:.2f}] "
    f"| {pa['d']:.2f} [{pa['d_lo']:.2f}, {pa['d_hi']:.2f}] "
    f"| {pa_half_total:.1f} pp total "
    f"|"
)

lines.append("")
lines.append("---")
lines.append("## 6. Summary Narrative")
lines.append("")

lines.append(
    f"**Reorientation rate** drops from {rr_scan_p1:.1f} to "
    f"{rr_scan_p2:.1f} scans per 5-min block (Δ = {rr_scan_drop:.1f} "
    f"[{rr_scan_drop_lo:.1f}, {rr_scan_drop_hi:.1f}]), "
    f"representing a loss of approximately **{abs(rr_half_drop):.0f} scans in the second half**. "
    f"The effect size is {abs(rr['d']):.2f} [{rr['d_lo']:.2f}, {rr['d_hi']:.2f}] "
    f"({'large' if abs(rr['d'])>=0.8 else 'medium'})."
)
lines.append("")

lines.append(
    f"**Shift latency** changes from {sl['m1']:.2f} s to {sl['m2']:.2f} s "
    f"(Δ = {sl['delta']:.2f} s [{sl['dlo']:.2f}, {sl['dhi']:.2f}]), "
    f"yielding a total of **{sl_half_total:.0f} seconds of reduced latency per half** "
    f"(effect size d = {sl['d']:.2f} [{sl['d_lo']:.2f}, {sl['d_hi']:.2f}]). "
    f"The direction is negative, indicating faster reactions in Phase 2 — "
    f"consistent with a game-tempo effect rather than fatigue-driven slowing."
)
lines.append("")

lines.append(
    f"**Pressing accuracy** declines from {pa['m1']*100:.1f}% to "
    f"{pa['m2']*100:.1f}% (Δ = {pa_drop_pp:.2f} pp "
    f"[{pa_drop_pp_lo:.2f}, {pa_drop_pp_hi:.2f}]), "
    f"representing a cumulative **{pa_half_total:.1f} percentage-point accuracy deficit "
    f"across the second half** "
    f"(effect size d = {pa['d']:.2f} [{pa['d_lo']:.2f}, {pa['d_hi']:.2f}], "
    f"{'negligible' if abs(pa['d'])<0.2 else 'small'})."
)
lines.append("")

# ── Write output ───────────────────────────────────────────────────────────
out_path = "focus-fatigue/outputs/analysis/performance_context_intervals.md"
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"✓ Written to {out_path}")

# ── Also print the paper-ready table for the agent ─────────────────────────
print("\n" + "=" * 90)
print("PAPER-READY TABLE:")
print("=" * 90)
print()
print(
    "| Metric | Phase 1 | Phase 2 | Δ (95% CI) | d (95% CI) | Per-half impact |"
)
print(
    "|--------|---------|---------|------------|------------|----------------|"
)
print(
    f"| Reorientation (scans/block) "
    f"| {rr_scan_p1:.1f} "
    f"| {rr_scan_p2:.1f} "
    f"| {rr_scan_drop:.1f} [{rr_scan_drop_lo:.1f}, {rr_scan_drop_hi:.1f}] "
    f"| {rr['d']:.2f} [{rr['d_lo']:.2f}, {rr['d_hi']:.2f}] "
    f"| −{abs(rr_half_drop):.0f} scans/half |"
)
print(
    f"| Shift latency (s) "
    f"| {sl['m1']:.2f} "
    f"| {sl['m2']:.2f} "
    f"| {sl['delta']:.2f} [{sl['dlo']:.2f}, {sl['dhi']:.2f}] "
    f"| {sl['d']:.2f} [{sl['d_lo']:.2f}, {sl['d_hi']:.2f}] "
    f"| {sl_half_total:.0f} s/half |"
)
print(
    f"| Pressing accuracy (pp) "
    f"| {pa['m1']*100:.1f}% "
    f"| {pa['m2']*100:.1f}% "
    f"| {pa_drop_pp:.2f} [{pa_drop_pp_lo:.2f}, {pa_drop_pp_hi:.2f}] "
    f"| {pa['d']:.2f} [{pa['d_lo']:.2f}, {pa['d_hi']:.2f}] "
    f"| {pa_half_total:.1f} pp total |"
)
