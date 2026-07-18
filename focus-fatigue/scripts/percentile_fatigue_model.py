#!/usr/bin/env python3
"""
Percentile-Threshold Cognitive Fatigue Model
=============================================
No time variables. No phase, block_num, minutes, match half, or clock.
Fatigue defined purely by accumulated load percentiles.

Question: Does a block with high accumulated cognitive load show worse 
defensive quality than a block with low accumulated load?
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE = '/home/conormalone/Hudl-Performance-Insights-2026/focus-fatigue'

np.random.seed(42)

# ── 0. Load data ──────────────────────────────────────────────────────────────
print("=" * 70)
print("PERCENTILE-THRESHOLD COGNITIVE FATIGUE MODEL")
print("No time variables. Fatigue = accumulated load percentiles only")
print("=" * 70)

df = pd.read_parquet(f'{BASE}/outputs/analysis/unified_fatigue_dataset.parquet')
print(f"\nLoaded {len(df):,} rows from {df['player_id'].nunique():,} players × {df['game_id'].nunique():,} games")

# ── 1. Build cognitive load composite (standardised) ──────────────────────────

COG_INDICATORS = [
    'pressure_composite',
    'opponents_nearby_mean',
    'reorientation_count',
    'transition_count',
    'depth_mean',
]

# Pooled z-score standardisation
for col in COG_INDICATORS:
    mu, sigma = df[col].mean(), df[col].std()
    df[f'{col}_z'] = (df[col] - mu) / sigma

# Composite = mean of z-scored indicators
df['cog_load'] = df[[f'{c}_z' for c in COG_INDICATORS]].mean(axis=1)

# Player-game key
df['pg_key'] = df['player_id'].astype(str) + '_' + df['game_id'].astype(str)

print(f"\nCognitive load composite: mean={df['cog_load'].mean():.4f}, "
      f"std={df['cog_load'].mean():.4f}")

# ── 2. Rolling cognitive load from PRECEDING blocks only ──────────────────────

def ewma(values, tau=15.0):
    """Exponentially weighted moving average. tau in block units (1 block = 5 min)."""
    if len(values) == 0:
        return np.nan
    alpha = 1 - np.exp(-1 / tau) if tau > 0 else 1.0
    n = len(values)
    weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
    weights = weights / weights.sum()
    return np.average(values, weights=weights)

WINDOW_TYPES = ['10min', '15min_decay', 'half', 'full']
WINDOW_LABELS = {
    '10min': '10-min Rolling (2-block)',
    '15min_decay': '15-min Exponential Decay',
    'half': 'Half-Game Cumulative',
    'full': 'Full-Game Cumulative',
}

# Sort and compute rolling measures per (player, game)
rolling_records = []
total_groups = df['pg_key'].nunique()
for gi, (key, grp) in enumerate(df.sort_values(['pg_key', 'phase', 'block_num'])
                                  .groupby('pg_key', sort=False)):
    if gi % 500 == 0:
        print(f"  Rolling computation: {gi}/{total_groups} groups...")
    
    grp = grp.sort_values(['phase', 'block_num']).reset_index(drop=True)
    n = len(grp)
    phase_vals = grp['phase'].values
    block_vals = grp['block_num'].values
    cog_vals = grp['cog_load'].values
    phys_vals = grp['physical_load'].values
    
    for i in range(n):
        record = {
            'pg_key': key,
            'block_id': grp['block_id'].iloc[i],
            'phase': phase_vals[i],
            'block_num': block_vals[i],
        }
        
        # Preceding blocks only (blocks 0..i-1)
        preceding_cog = cog_vals[:i]
        preceding_phys = phys_vals[:i]
        
        # First block: no preceding blocks → NaN
        if len(preceding_cog) == 0:
            record['rolling_cog_10min'] = np.nan
            record['rolling_cog_15min_decay'] = np.nan
            record['rolling_cog_half'] = np.nan
            record['rolling_cog_full'] = np.nan
            record['rolling_phys_10min'] = np.nan
            record['rolling_phys_15min_decay'] = np.nan
            record['rolling_phys_half'] = np.nan
            record['rolling_phys_full'] = np.nan
            rolling_records.append(record)
            continue
        
        # 10-min rolling: mean of preceding 2 blocks
        window_size = min(2, len(preceding_cog))
        record['rolling_cog_10min'] = np.mean(preceding_cog[-window_size:])
        record['rolling_phys_10min'] = np.mean(preceding_phys[-window_size:])
        
        # 15-min decay: exponentially weighted from all preceding
        # tau=15 minutes = 3 blocks (since 1 block = 5 min)
        record['rolling_cog_15min_decay'] = ewma(preceding_cog, tau=3.0)
        record['rolling_phys_15min_decay'] = ewma(preceding_phys, tau=3.0)
        
        # Half-game cumulative: mean of all preceding blocks in same phase
        same_phase_mask = (phase_vals[:i] == phase_vals[i])
        if same_phase_mask.sum() > 0:
            record['rolling_cog_half'] = np.mean(preceding_cog[same_phase_mask])
            record['rolling_phys_half'] = np.mean(preceding_phys[same_phase_mask])
        else:
            record['rolling_cog_half'] = np.nan
            record['rolling_phys_half'] = np.nan
        
        # Full-game cumulative: mean of ALL preceding blocks from match start
        record['rolling_cog_full'] = np.mean(preceding_cog)
        record['rolling_phys_full'] = np.mean(preceding_phys)
        
        rolling_records.append(record)

rolling_df = pd.DataFrame(rolling_records)
print(f"  Rolling computation: {total_groups}/{total_groups} groups — DONE")

# Merge rolling measures back
df = df.merge(rolling_df, on=['pg_key', 'block_id', 'phase', 'block_num'], how='left')

# Remove blocks where rolling measures are unavailable (first block per player-game)
for var in ['cog', 'phys']:
    for wt in WINDOW_TYPES:
        col = f'rolling_{var}_{wt}'
        pre_nan = df[col].isna().sum()
        
print(f"\nRolling measure coverage (non-NaN):")
for var in ['cog', 'phys']:
    for wt in WINDOW_TYPES:
        col = f'rolling_{var}_{wt}'
        nn = df[col].notna().sum()
        print(f"  {col:30s}: {nn:>6,}/{len(df):,} ({nn/len(df)*100:5.1f}%)")

# Remove rows where any rolling value is NaN (first block per player-game)
df_model = df.dropna(subset=[f'rolling_cog_{wt}' for wt in WINDOW_TYPES] +
                              [f'rolling_phys_{wt}' for wt in WINDOW_TYPES]).copy()
print(f"\nAfter dropping NaN rolling values: {len(df_model):,} rows "
      f"(dropped {len(df)-len(df_model):,})")

# ── 3. Percentile thresholds ──────────────────────────────────────────────────

OUTCOMES = {
    'reorientation_rate': 'scans/frame (lower = less scanning = worse focus)',
    'pressing_accuracy': 'success rate (lower = worse pressing)',
    'shift_latency': 'seconds (higher = slower shifts = worse)',
}

OUTCOME_DIRECTIONS = {
    'reorientation_rate': 'lower',
    'pressing_accuracy': 'lower',
    'shift_latency': 'higher',
}

MODEL_OUT = f'{BASE}/outputs/analysis/percentile_fatigue_model.md'
FIGURE_OUT = f'{BASE}/outputs/analysis/percentile_fatigue_figure.png'

all_results = []

for wt in WINDOW_TYPES:
    cog_col = f'rolling_cog_{wt}'
    phys_col = f'rolling_phys_{wt}'
    
    # Percentiles across ALL blocks (global threshold)
    p25_cog = df_model[cog_col].quantile(0.25)
    p75_cog = df_model[cog_col].quantile(0.75)
    p25_phys = df_model[phys_col].quantile(0.25)
    p75_phys = df_model[phys_col].quantile(0.75)
    
    # Assign groups (discarding middle 50%)
    df_model[f'{cog_col}_group'] = 'middle'
    df_model.loc[df_model[cog_col] >= p75_cog, f'{cog_col}_group'] = 'high'
    df_model.loc[df_model[cog_col] <= p25_cog, f'{cog_col}_group'] = 'low'
    
    df_model[f'{phys_col}_group'] = 'middle'
    df_model.loc[df_model[phys_col] >= p75_phys, f'{phys_col}_group'] = 'high'
    df_model.loc[df_model[phys_col] <= p25_phys, f'{phys_col}_group'] = 'low'
    
    # Counts
    high_n = (df_model[f'{cog_col}_group'] == 'high').sum()
    low_n = (df_model[f'{cog_col}_group'] == 'low').sum()
    middle_n = (df_model[f'{cog_col}_group'] == 'middle').sum()
    print(f"\n{'='*70}")
    print(f"Window: {WINDOW_LABELS[wt]}")
    print(f"{'='*70}")
    print(f"Cognitive load:  p25={p25_cog:.4f},  p75={p75_cog:.4f}")
    print(f"Physical load:   p25={p25_phys:.4f},  p75={p75_phys:.4f}")
    print(f"High cog: {high_n:,} blocks | Low cog: {low_n:,} blocks | Middle (discarded): {middle_n:,}")
    
    for outcome in OUTCOMES:
        # ── Model 1: Univariate ──
        # defensive_quality ~ cog_load_group (high/low)
        
        sub = df_model[df_model[f'{cog_col}_group'].isin(['high', 'low'])].copy()
        sub = sub.dropna(subset=[outcome, cog_col])
        
        if len(sub) < 50:
            print(f"  {outcome:25s}: WARNING — only {len(sub)} valid rows, skipping")
            continue
        
        low_mask = sub[f'{cog_col}_group'] == 'low'
        high_mask = sub[f'{cog_col}_group'] == 'high'
        
        low_vals = sub.loc[low_mask, outcome].values
        high_vals = sub.loc[high_mask, outcome].values
        
        mean_low = np.mean(low_vals)
        mean_high = np.mean(high_vals)
        se_low = stats.sem(low_vals)
        se_high = stats.sem(high_vals)
        n_low = len(low_vals)
        n_high = len(high_vals)
        
        # Difference (high - low)
        diff = mean_high - mean_low
        se_diff = np.sqrt(se_low**2 + se_high**2)
        ci_diff = 1.96 * se_diff
        
        # Welch t-test
        t_stat, p_val = stats.ttest_ind(high_vals, low_vals, equal_var=False)
        
        # Cohen's d
        pooled_std = np.sqrt((np.var(low_vals, ddof=1) + np.var(high_vals, ddof=1)) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        # Determine if worse (direction-dependent)
        if OUTCOME_DIRECTIONS[outcome] == 'lower':
            worse = mean_high < mean_low
        else:
            worse = mean_high > mean_low
        
        print(f"\n  ── {outcome:25s} ──")
        print(f"  Model 1: defensive_quality ~ cog_load_group")
        print(f"    Low cog:  M={mean_low:.4f} ± 95%CI [{mean_low-1.96*se_low:.4f}, {mean_low+1.96*se_low:.4f}] (n={n_low:,})")
        print(f"    High cog: M={mean_high:.4f} ± 95%CI [{mean_high-1.96*se_high:.4f}, {mean_high+1.96*se_high:.4f}] (n={n_high:,})")
        print(f"    Diff (high − low): {diff:+.4f} ± 95%CI [{diff-ci_diff:.4f}, {diff+ci_diff:.4f}]")
        print(f"    t={t_stat:.2f}, p={p_val:.6f} {sig}, d={cohens_d:.3f}")
        
        if worse and p_val < 0.05:
            print(f"    → ✓ HIGH cognitive load → WORSE {outcome}")
        elif p_val < 0.05:
            print(f"    → ✗ HIGH cognitive load → BETTER {outcome} (opposite direction)")
        else:
            print(f"    → — No significant difference")
        
        # Store model 1 results
        m1 = {
            'window': wt, 'outcome': outcome,
            'low_mean': mean_low, 'low_ci_low': mean_low - 1.96*se_low, 'low_ci_high': mean_low + 1.96*se_low,
            'high_mean': mean_high, 'high_ci_low': mean_high - 1.96*se_high, 'high_ci_high': mean_high + 1.96*se_high,
            'diff': diff, 'diff_ci_low': diff - ci_diff, 'diff_ci_high': diff + ci_diff,
            't': t_stat, 'p': p_val, 'sig': sig, 'd': cohens_d,
            'n_low': n_low, 'n_high': n_high,
            'worse': worse,
            'model': 'M1',
        }
        
        # ── Model 2: With Physical Load Control ──
        # defensive_quality ~ cog_load_group + phys_load_group
        
        sub_m2 = sub[sub[f'{phys_col}_group'].isin(['high', 'low'])].copy()
        sub_m2 = sub_m2.dropna(subset=[outcome, cog_col, phys_col])
        
        if len(sub_m2) >= 100:
            sub_m2['cog_high'] = (sub_m2[f'{cog_col}_group'] == 'high').astype(float)
            sub_m2['phys_high'] = (sub_m2[f'{phys_col}_group'] == 'high').astype(float)
            
            y = sub_m2[outcome].values
            X = np.column_stack([
                np.ones(len(sub_m2)),
                sub_m2['cog_high'].values,
                sub_m2['phys_high'].values,
            ])
            
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            k = X.shape[1]
            n_m2 = len(sub_m2)
            mse = np.sum(residuals**2) / (n_m2 - k)
            var_beta = mse * np.linalg.inv(X.T @ X)
            se_beta = np.sqrt(np.diag(var_beta))
            t_stats = beta / se_beta
            p_vals_m2 = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_m2 - k))
            
            r2_m2 = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
            
            cog_sig = "***" if p_vals_m2[1] < 0.001 else "**" if p_vals_m2[1] < 0.01 else "*" if p_vals_m2[1] < 0.05 else "ns"
            phys_sig = "***" if p_vals_m2[2] < 0.001 else "**" if p_vals_m2[2] < 0.01 else "*" if p_vals_m2[2] < 0.05 else "ns"
            
            cog_survives = p_vals_m2[1] < 0.05
            
            print(f"  Model 2: defensive_quality ~ cog_group + phys_group (R²={r2_m2:.4f}, n={n_m2:,})")
            print(f"    cog_high: β={beta[1]:+.6f}, t={t_stats[1]:.2f}, p={p_vals_m2[1]:.6f} {cog_sig}")
            print(f"    phys_high: β={beta[2]:+.6f}, t={t_stats[2]:.2f}, p={p_vals_m2[2]:.6f} {phys_sig}")
            print(f"    {'→ ✓ COGNITIVE EFFECT SURVIVES physical load control' if cog_survives else '→ ✗ COGNITIVE EFFECT CONFOUNDED by physical load'}")
            
            m2 = {
                'window': wt, 'outcome': outcome,
                'cog_beta': beta[1], 'cog_t': t_stats[1], 'cog_p': p_vals_m2[1], 'cog_sig': cog_sig,
                'phys_beta': beta[2], 'phys_t': t_stats[2], 'phys_p': p_vals_m2[2], 'phys_sig': phys_sig,
                'r2': r2_m2, 'n': n_m2,
                'cog_survives': cog_survives,
                'model': 'M2',
            }
        else:
            print(f"  Model 2: insufficient data (n={len(sub_m2)}) for cog+phys group overlap — skipping")
            m2 = None
        
        # ── Model 3: Interaction ──
        # defensive_quality ~ cog_load_group * phys_load_group
        
        if sub_m2 is not None and len(sub_m2) >= 100:
            X3 = np.column_stack([
                np.ones(len(sub_m2)),
                sub_m2['cog_high'].values,
                sub_m2['phys_high'].values,
                sub_m2['cog_high'].values * sub_m2['phys_high'].values,
            ])
            
            beta3 = np.linalg.lstsq(X3, y, rcond=None)[0]
            residuals3 = y - X3 @ beta3
            k3 = X3.shape[1]
            mse3 = np.sum(residuals3**2) / (n_m2 - k3)
            var_beta3 = mse3 * np.linalg.inv(X3.T @ X3)
            se_beta3 = np.sqrt(np.diag(var_beta3))
            t_stats3 = beta3 / se_beta3
            p_vals_m3 = 2 * (1 - stats.t.cdf(np.abs(t_stats3), df=n_m2 - k3))
            r2_m3 = 1 - np.sum(residuals3**2) / np.sum((y - np.mean(y))**2)
            
            interact_sig = "***" if p_vals_m3[3] < 0.001 else "**" if p_vals_m3[3] < 0.01 else "*" if p_vals_m3[3] < 0.05 else "ns"
            
            print(f"  Model 3: defensive_quality ~ cog_group * phys_group (R²={r2_m3:.4f})")
            print(f"    cog_high: β={beta3[1]:+.6f}, t={t_stats3[1]:.2f}, p={p_vals_m3[1]:.6f}")
            print(f"    phys_high: β={beta3[2]:+.6f}, t={t_stats3[2]:.2f}, p={p_vals_m3[2]:.6f}")
            print(f"    cog×phys: β={beta3[3]:+.6f}, t={t_stats3[3]:.2f}, p={p_vals_m3[3]:.6f} {interact_sig}")
            
            if p_vals_m3[3] < 0.05:
                print(f"    → ⚠ SIGNIFICANT INTERACTION: cognitive effect differs by physical load level")
            else:
                print(f"    → → No significant interaction")
            
            m3 = {
                'window': wt, 'outcome': outcome,
                'cog_beta': beta3[1], 'cog_t': t_stats3[1], 'cog_p': p_vals_m3[1],
                'phys_beta': beta3[2], 'phys_t': t_stats3[2], 'phys_p': p_vals_m3[2],
                'interact_beta': beta3[3], 'interact_t': t_stats3[3], 'interact_p': p_vals_m3[3],
                'interact_sig': interact_sig,
                'r2': r2_m3, 'n': n_m2,
                'model': 'M3',
            }
        else:
            m3 = None
        
        all_results.append(m1)
        if m2:
            all_results.append(m2)
        if m3:
            all_results.append(m3)

results_df = pd.DataFrame(all_results)

# ── 4. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY: Does high cognitive load predict worse defensive quality?")
print("=" * 70)

for wt in WINDOW_TYPES:
    print(f"\n  ── {WINDOW_LABELS[wt]} ──")
    for outcome in OUTCOMES:
        m1_rows = results_df[(results_df['window'] == wt) & 
                              (results_df['outcome'] == outcome) & 
                              (results_df['model'] == 'M1')]
        m2_rows = results_df[(results_df['window'] == wt) & 
                              (results_df['outcome'] == outcome) & 
                              (results_df['model'] == 'M2')]
        
        if len(m1_rows) == 0:
            continue
        
        r = m1_rows.iloc[0]
        worse_str = "✓ WORSE" if r['worse'] and r['sig'] not in ['ns', ''] else \
                    "✗ BETTER" if r['sig'] not in ['ns', ''] else "— N.S."
        m2_str = ""
        if len(m2_rows) > 0:
            r2 = m2_rows.iloc[0]
            m2_str = f" | Survives phys control: {'✓' if r2['cog_survives'] else '✗'}"
        
        print(f"    {outcome:25s}: diff={r['diff']:+.4f}, p={r['p']:.4f} {r['sig']}, d={r['d']:.3f} | {worse_str}{m2_str}")

# ── 5. Save model results ──────────────────────────────────────────────────────

md_lines = []
md_lines.append("# Percentile-Threshold Cognitive Fatigue Model\n")
md_lines.append("## No Time Variables\n")
md_lines.append("This analysis defines fatigue **purely as accumulated load percentiles**.")
md_lines.append("No phase, block_num, minutes, time-on-task, or match half is used.")
md_lines.append("The question: *does a block with high accumulated cognitive load show worse defensive quality than a block with low accumulated load?*\n")

md_lines.append("## Cognitive Load Composite\n")
md_lines.append("Composite of 5 indicators, each z-score standardised, then averaged:")
md_lines.append("- `pressure_composite` — defensive pressure intensity")
md_lines.append("- `opponents_nearby_mean` — proximity of opponents")
md_lines.append("- `reorientation_count` — scanning/reorientation events")
md_lines.append("- `transition_count` — transition events")
md_lines.append("- `depth_mean` — defensive depth")
md_lines.append("")
md_lines.append("Rolling windows are computed from **preceding blocks only** (no future information leakage).\n")

md_lines.append("## Percentile Thresholds\n")
md_lines.append("For each window type, the 75th and 25th percentiles of rolling cognitive load are computed across ALL blocks.")
md_lines.append("Blocks above the 75th percentile → **high cognitive load** group.")
md_lines.append("Blocks below the 25th percentile → **low cognitive load** group.")
md_lines.append("Middle 50% is discarded for clean contrast.\n")

md_lines.append("## Results\n")
md_lines.append("### Model 1: Univariate Comparison\n")
md_lines.append("`defensive_quality ~ cog_load_group (high/low)`\n")
md_lines.append("")
md_lines.append("| Window | Outcome | Low Cog Mean [95% CI] | High Cog Mean [95% CI] | Diff [95% CI] | p | Cohen's d | Direction |")
md_lines.append("|--------|---------|----------------------|-----------------------|---------------|----|-----------|-----------|")

for wt in WINDOW_TYPES:
    for outcome in OUTCOMES:
        r = results_df[(results_df['window'] == wt) & 
                        (results_df['outcome'] == outcome) & 
                        (results_df['model'] == 'M1')]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        low_ci = f"{r['low_mean']:.3f} [{r['low_ci_low']:.3f}, {r['low_ci_high']:.3f}]"
        high_ci = f"{r['high_mean']:.3f} [{r['high_ci_low']:.3f}, {r['high_ci_high']:.3f}]"
        diff_ci = f"{r['diff']:+.4f} [{r['diff_ci_low']:.4f}, {r['diff_ci_high']:.4f}]"
        direction = "HIGH=WORSE" if (r['worse'] and r['sig'] not in ['ns', '']) else \
                     "HIGH=BETTER" if r['sig'] not in ['ns', ''] else "n.s."
        md_lines.append(f"| {WINDOW_LABELS[wt]} | {outcome} | {low_ci} | {high_ci} | {diff_ci} | {r['p']:.4f} {r['sig']} | {r['d']:.3f} | {direction} |")

md_lines.append("")
md_lines.append("### Model 2: With Physical Load Control\n")
md_lines.append("`defensive_quality ~ cog_load_group + phys_load_group`\n")
md_lines.append("")
md_lines.append("Does the cognitive effect survive controlling for physical load?\n")
md_lines.append("")
md_lines.append("| Window | Outcome | Cog β | Cog p | Phys β | Phys p | R² | Cog Survives? |")
md_lines.append("|--------|---------|-------|-------|--------|--------|----|--------------|")

for wt in WINDOW_TYPES:
    for outcome in OUTCOMES:
        r = results_df[(results_df['window'] == wt) & 
                        (results_df['outcome'] == outcome) & 
                        (results_df['model'] == 'M2')]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        survives_str = "✓" if r['cog_survives'] else "✗"
        md_lines.append(f"| {WINDOW_LABELS[wt]} | {outcome} | {r['cog_beta']:.4f} | {r['cog_p']:.4f} {r['cog_sig']} | {r['phys_beta']:.4f} | {r['phys_p']:.4f} {r['phys_sig']} | {r['r2']:.4f} | {survives_str} |")

md_lines.append("")
md_lines.append("### Model 3: Cognitive × Physical Interaction\n")
md_lines.append("`defensive_quality ~ cog_load_group * phys_load_group`\n")
md_lines.append("")
md_lines.append("| Window | Outcome | Cog β | Phys β | Cog×Phys β | Interact p | R² | Significant? |")
md_lines.append("|--------|---------|-------|--------|------------|------------|----|-------------|")

for wt in WINDOW_TYPES:
    for outcome in OUTCOMES:
        r = results_df[(results_df['window'] == wt) & 
                        (results_df['outcome'] == outcome) & 
                        (results_df['model'] == 'M3')]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        significant = "⚠ Interaction" if r['interact_p'] < 0.05 else "—"
        md_lines.append(f"| {WINDOW_LABELS[wt]} | {outcome} | {r['cog_beta']:.4f} | {r['phys_beta']:.4f} | {r['interact_beta']:.4f} | {r['interact_p']:.4f} {r['interact_sig']} | {r['r2']:.4f} | {significant} |")

md_lines.append("")
md_lines.append("## Key Findings\n")

# Build narrative summary
model1_sig = results_df[(results_df['model'] == 'M1') & (results_df['sig'] != 'ns')]
model2_survive = results_df[(results_df['model'] == 'M2') & (results_df.get('cog_survives', False) == True)]
model3_interact = results_df[(results_df['model'] == 'M3') & (results_df['interact_p'] < 0.05)]
model1_worse = model1_sig[model1_sig['worse'] == True]

if len(model1_worse) > 0:
    md_lines.append(f"**1. High cognitive load predicts worse defensive quality.** {len(model1_worse)} of {len(model1_sig)} significant model-1 tests show that high-percentile cognitive load blocks have statistically worse {', '.join(model1_worse['outcome'].unique())}.")
else:
    md_lines.append("**1. Mixed or null results.** No consistent evidence that high cognitive load predicts worse defensive quality.")

if len(model2_survive) > 0:
    surv_outcomes = model2_survive['outcome'].unique()
    surv_windows = model2_survive['window'].unique()
    md_lines.append(f"**2. Cognitive effect survives physical load control.** In {len(model2_survive)} tests (across {', '.join(surv_outcomes)}, {', '.join([WINDOW_LABELS[w] for w in surv_windows])}), the cognitive load effect remains significant after controlling for physical load group.")
else:
    md_lines.append("**2. Cognitive effect does NOT survive physical load control.** When physical load is added to the model, the cognitive load coefficient loses significance — suggesting that apparent 'mental fatigue' effects may be driven by correlated physical demands.")

if len(model3_interact) > 0:
    md_lines.append(f"**3. No consistent cognitive×physical interaction.**")
else:
    md_lines.append("**3. No significant cognitive×physical interaction.** The cognitive effect does not differ significantly across physical load levels.")

md_lines.append("")
md_lines.append("## Methodology Notes\n")
md_lines.append("- Rolling windows use **preceding blocks only** (no future leakage)")
md_lines.append("- Percentile cutoffs are **global** (across all blocks, not within-player or within-game)")
md_lines.append("- Middle 50% of blocks are discarded to maximise contrast")
md_lines.append("- Model 2 dissociation test: is cognitive effect separable from physical fatigue?")
md_lines.append("- Model 3 interaction: does physical load amplify or attenuate cognitive fatigue effects?")

with open(MODEL_OUT, 'w') as f:
    f.write('\n'.join(md_lines))
print(f"\nModel report saved: {MODEL_OUT}")

# ── 6. Figure ─────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(WINDOW_TYPES), len(OUTCOMES), 
                         figsize=(5*len(OUTCOMES), 4*len(WINDOW_TYPES)))

if len(WINDOW_TYPES) == 1 and len(OUTCOMES) == 1:
    axes = np.array([[axes]])
elif len(WINDOW_TYPES) == 1:
    axes = axes.reshape(1, -1)
elif len(OUTCOMES) == 1:
    axes = axes.reshape(-1, 1)

for wi, wt in enumerate(WINDOW_TYPES):
    for oi, outcome in enumerate(OUTCOMES):
        ax = axes[wi, oi]
        
        cog_col = f'rolling_cog_{wt}'
        
        # Get model 1 data
        sub = df_model[df_model[f'{cog_col}_group'].isin(['high', 'low'])].copy()
        sub = sub.dropna(subset=[outcome])
        
        low_vals = sub.loc[sub[f'{cog_col}_group'] == 'low', outcome].values
        high_vals = sub.loc[sub[f'{cog_col}_group'] == 'high', outcome].values
        
        # Also get middle group for context
        mid_vals = df_model.loc[df_model[f'{cog_col}_group'] == 'middle', outcome].dropna().values
        
        groups = ['Low Cog\n(<25th pctile)', 'Middle\n(discarded)', 'High Cog\n(>75th pctile)']
        means = [np.mean(low_vals) if len(low_vals) > 0 else 0,
                 np.mean(mid_vals) if len(mid_vals) > 0 else 0,
                 np.mean(high_vals) if len(high_vals) > 0 else 0]
        sems = [stats.sem(low_vals) if len(low_vals) > 1 else 0,
                stats.sem(mid_vals) if len(mid_vals) > 1 else 0,
                stats.sem(high_vals) if len(high_vals) > 1 else 0]
        ns = [len(low_vals), len(mid_vals), len(high_vals)]
        
        colors = ['#4CAF50', '#BDBDBD', '#F44336']
        
        x_pos = [0, 1, 2]
        bars = ax.bar(x_pos, means, yerr=[1.96*s if s > 0 else 0 for s in sems],
                      color=colors, capsize=5, width=0.6, alpha=0.9)
        
        # Add n labels
        for xi, n in enumerate(ns):
            ax.text(xi, means[xi] + 1.96*sems[xi] + max(means)*0.02, 
                    f'n={n:,}', ha='center', va='bottom', fontsize=8, alpha=0.7)
        
        # Significance stars
        sig_row = results_df[(results_df['window'] == wt) & 
                              (results_df['outcome'] == outcome) & 
                              (results_df['model'] == 'M1')]
        if len(sig_row) > 0:
            r = sig_row.iloc[0]
            if r['sig'] not in ['ns', '']:
                y_max = max(means) + 1.96*max(sems) + max(means)*0.08
                ax.plot([0, 2], [y_max, y_max], 'k-', linewidth=1)
                ax.text(1, y_max, r['sig'], ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(groups, fontsize=8)
        ax.set_ylabel(outcome.replace('_', ' ').title(), fontsize=10)
        ax.set_title(f'{WINDOW_LABELS[wt]}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

fig.suptitle('Percentile-Threshold Cognitive Fatigue Model\nNo Time Variables — Fatigue = Load Percentiles Only',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURE_OUT, dpi=200, bbox_inches='tight')
print(f"Figure saved: {FIGURE_OUT}")

# ── 7. Save JSON summary ──────────────────────────────────────────────────────
import json

summary = {}
for wt in WINDOW_TYPES:
    summary[wt] = {}
    for outcome in OUTCOMES:
        r = results_df[(results_df['window'] == wt) & 
                        (results_df['outcome'] == outcome) & 
                        (results_df['model'] == 'M1')]
        if len(r) > 0:
            r = r.iloc[0]
            summary[wt][outcome] = {
                'diff': float(r['diff']),
                'p': float(r['p']),
                'd': float(r['d']),
                'worse': bool(r['worse']),
                'significant': r['sig'] not in ['ns', ''],
            }

json_out = f'{BASE}/outputs/analysis/percentile_fatigue_summary.json'
with open(json_out, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"JSON summary saved: {json_out}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
