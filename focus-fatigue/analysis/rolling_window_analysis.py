"""
Rolling Window Fatigue Analysis — Proper Dissociation
=====================================================
Computes windowed measures for physical load and cognitive signals,
then tests whether cognitive fatigue effects survive controlling for
physical fatigue on the same window.
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

# ── Config ──
TAU = 15.0
BLOCK_MINUTES = 5.0
COGNITIVE_SIGNALS = [
    'reorientation_rate', 'pressing_accuracy', 'shift_latency',
    'positional_drift', 'transition_latency',
]
WINDOW_TYPES = ['10min_rolling', '15min_decaying', 'half_game', 'full_game']

# ── Load ──
log("Loading data...")
df = pd.read_parquet('focus-fatigue/outputs/analysis/unified_fatigue_dataset.parquet')
df = df.sort_values(['player_id', 'game_id', 'phase', 'block_num']).reset_index(drop=True)
df['block_global'] = df.groupby(['player_id', 'game_id']).cumcount()
log(f"Loaded {len(df)} rows, {df['player_id'].nunique()} players, {df['game_id'].nunique()} games")

# ── Compute windowed measures ──
log("Computing windowed measures (looping over player×game groups)...")

# Pre-allocate windowed columns
sig_cols_win = []
for wtype in WINDOW_TYPES:
    df[f'phys_{wtype}'] = np.nan
    for sig in COGNITIVE_SIGNALS:
        col = f'{sig}_{wtype}'
        df[col] = np.nan
        sig_cols_win.append(col)

groups = list(df.groupby(['player_id', 'game_id']))
n_groups = len(groups)

for gi, ((pid, gid), gdf) in enumerate(groups):
    if gi % 300 == 0:
        log(f"  Processing group {gi}/{n_groups}...")
    
    idx = gdf.index.values
    g = gdf.sort_values('block_global')
    g_idx = g.index.values
    n = len(g)
    
    phys_vals = g['physical_load'].values
    sig_vals = {}
    for sig in COGNITIVE_SIGNALS:
        sig_vals[sig] = g[sig].values
    phases = g['phase'].values
    
    # 1) 10-min rolling (lagging)
    for i in range(n):
        if i == 0:
            window_phys = phys_vals[0:1]
            window_sigs = {s: sig_vals[s][0:1] for s in COGNITIVE_SIGNALS}
        else:
            window_phys = phys_vals[i-1:i+1]
            window_sigs = {s: sig_vals[s][i-1:i+1] for s in COGNITIVE_SIGNALS}
        
        df.loc[g_idx[i], 'phys_10min_rolling'] = window_phys.mean()
        for sig in COGNITIVE_SIGNALS:
            valid = window_sigs[sig][~np.isnan(window_sigs[sig])]
            if len(valid) >= 1:
                df.loc[g_idx[i], f'{sig}_10min_rolling'] = valid.mean()
    
    # 2) 15-min decaying (exponential)
    for i in range(n):
        lags = np.arange(i, -1, -1) * BLOCK_MINUTES
        weights = np.exp(-lags / TAU)
        weights = weights / weights.sum()
        
        df.loc[g_idx[i], f'phys_15min_decaying'] = np.average(phys_vals[:i+1], weights=weights)
        for sig in COGNITIVE_SIGNALS:
            vals = sig_vals[sig][:i+1]
            mask = ~np.isnan(vals)
            if mask.sum() >= 1:
                w = weights[mask] / weights[mask].sum()
                df.loc[g_idx[i], f'{sig}_15min_decaying'] = np.average(vals[mask], weights=w)
    
    # 3) Half-game
    for pval in [1, 2]:
        p_mask = phases == pval
        if p_mask.sum() == 0:
            continue
        phys_half = phys_vals[p_mask].mean()
        sig_half = {}
        for sig in COGNITIVE_SIGNALS:
            v = sig_vals[sig][p_mask]
            v_valid = v[~np.isnan(v)]
            sig_half[sig] = v_valid.mean() if len(v_valid) > 0 else np.nan
        
        mask_idx = g_idx[p_mask]
        df.loc[mask_idx, 'phys_half_game'] = phys_half
        for sig in COGNITIVE_SIGNALS:
            df.loc[mask_idx, f'{sig}_half_game'] = sig_half[sig]
    
    # 4) Full-game
    phys_full = phys_vals.mean()
    sig_full = {}
    for sig in COGNITIVE_SIGNALS:
        v = sig_vals[sig]
        v_valid = v[~np.isnan(v)]
        sig_full[sig] = v_valid.mean() if len(v_valid) > 0 else np.nan
    
    df.loc[g_idx, 'phys_full_game'] = phys_full
    for sig in COGNITIVE_SIGNALS:
        df.loc[g_idx, f'{sig}_full_game'] = sig_full[sig]

log("Windowed measures computed.")

# ── Verification ──
log("\nWindow verification:")
cols_verify = ['player_id','game_id','phase','block_global','physical_load','reorientation_rate', 'reorientation_rate_10min_rolling', 'reorientation_rate_15min_decaying', 'reorientation_rate_half_game', 'reorientation_rate_full_game']
print(df[cols_verify].head(10).to_string())

# Count non-null windowed values
for sig in COGNITIVE_SIGNALS:
    for wtype in WINDOW_TYPES:
        nn = df[f'{sig}_{wtype}'].notna().sum()
        tot = len(df)
        log(f"  {sig}_{wtype}: {nn}/{tot} non-null ({nn/tot*100:.1f}%)")

# ── Run regression models ──
results = []

for wtype in WINDOW_TYPES:
    log(f"\n{'='*60}")
    log(f"Window type: {wtype}")
    log('='*60)
    
    phys_col = f'phys_{wtype}'
    
    for sig in COGNITIVE_SIGNALS:
        sig_col = f'{sig}_{wtype}'
        
        sub = df[[sig_col, phys_col, 'phase']].dropna().copy()
        if len(sub) < 50:
            log(f"  {sig}: only {len(sub)} valid rows, skipping")
            continue
        
        sub['phase01'] = (sub['phase'] - 1).astype(float)
        
        # Model 1: univariate
        try:
            m1 = smf.ols(f'{sig_col} ~ phase01', data=sub).fit()
        except Exception as e:
            log(f"  {sig} M1 failed: {e}")
            continue
        
        # Model 2: controlled
        sub['phys_scaled'] = (sub[phys_col] - sub[phys_col].mean()) / sub[phys_col].std()
        try:
            m2 = smf.ols(f'{sig_col} ~ phase01 + phys_scaled', data=sub).fit()
        except Exception as e:
            log(f"  {sig} M2 failed: {e}")
            continue
        
        # Model 3: interaction
        try:
            m3 = smf.ols(f'{sig_col} ~ phase01 * phys_scaled', data=sub).fit()
        except Exception as e:
            log(f"  {sig} M3 failed: {e}")
            continue
        
        beta1 = m1.params.get('phase01', np.nan)
        p1 = m1.pvalues.get('phase01', np.nan)
        
        beta2_phase = m2.params.get('phase01', np.nan)
        p2_phase = m2.pvalues.get('phase01', np.nan)
        beta2_phys = m2.params.get('phys_scaled', np.nan)
        p2_phys = m2.pvalues.get('phys_scaled', np.nan)
        
        beta3_phase = m3.params.get('phase01', np.nan)
        p3_phase = m3.pvalues.get('phase01', np.nan)
        beta3_interact = m3.params.get('phase01:phys_scaled', np.nan)
        p3_interact = m3.pvalues.get('phase01:phys_scaled', np.nan)
        
        delta = beta2_phase - beta1 if not (np.isnan(beta1) or np.isnan(beta2_phase)) else np.nan
        if not np.isnan(beta1) and abs(beta1) > 1e-10:
            pct_attn = delta / beta1 * 100
        else:
            pct_attn = np.nan
        
        # Cohen's d for Model 1
        phase_means = sub.groupby('phase')[sig_col].mean()
        phase_stds = sub.groupby('phase')[sig_col].std()
        phase_counts = sub.groupby('phase')[sig_col].count()
        if 1 in phase_means.index and 2 in phase_means.index:
            pooled_sd = np.sqrt(
                ((phase_counts[1]-1)*phase_stds[1]**2 + (phase_counts[2]-1)*phase_stds[2]**2) /
                (phase_counts[1] + phase_counts[2] - 2)
            )
            cohens_d = (phase_means[2] - phase_means[1]) / pooled_sd if pooled_sd > 0 else np.nan
        else:
            cohens_d = np.nan
        
        survives = p2_phase < 0.05
        
        row = {
            'window': wtype,
            'signal': sig,
            'n_obs': len(sub),
            'phase_coef_m1': beta1,
            'phase_p_m1': p1,
            'phase_cohens_d_m1': cohens_d,
            'phase_coef_m2': beta2_phase,
            'phase_p_m2': p2_phase,
            'delta_phase': delta,
            'pct_attn': pct_attn,
            'survives_control': survives,
            'phys_coef_m2': beta2_phys,
            'phys_p_m2': p2_phys,
            'interact_coef_m3': beta3_interact,
            'interact_p_m3': p3_interact,
            'mean_phase1': phase_means.get(1, np.nan),
            'mean_phase2': phase_means.get(2, np.nan),
        }
        results.append(row)
        
        status = "✓ SURVIVES" if survives else "✗ CONFOUNDED"
        log(f"  {sig:25s} | M1 β={beta1:+.4f} p={p1:.4f} d={cohens_d:.3f} | "
            f"M2 β={beta2_phase:+.4f} p={p2_phase:.4f} | "
            f"Δ={delta:+.4f} ({pct_attn:+.0f}%) | "
            f"phys β={beta2_phys:+.4f} p={p2_phys:.4f} | "
            f"{status}")

results_df = pd.DataFrame(results)
results_df.to_csv('focus-fatigue/outputs/analysis/rolling_window_results_table.csv', index=False)

# ── Summary table ──
log("\n\n" + "="*60)
log("SUMMARY TABLE")
log("="*60)
log(f"{'Window':15s} | {'Signal':25s} | {'M1 β':8s} | {'M2 β':8s} | {'Δ':8s} | {'Attn%':6s} | {'Coh d':6s} | {'Phys β':7s} | {'Surv?':5s}")
log("-"*95)
for _, r in results_df.iterrows():
    surv = "✓" if r['survives_control'] else "✗"
    log(f"{r['window']:15s} | {r['signal']:25s} | {r['phase_coef_m1']:+8.4f} | {r['phase_coef_m2']:+8.4f} | "
        f"{r['delta_phase']:+8.4f} | {r['pct_attn']:+5.0f}% | {r['phase_cohens_d_m1']:+5.3f} | "
        f"{r['phys_coef_m2']:+7.4f} | {surv:5s}")

# ── Key finding ──
log("\n\n" + "="*60)
log("KEY FINDING: Cleanest dissociation")
log("="*60)

survivors = results_df[results_df['survives_control']].copy()
confounded = results_df[~results_df['survives_control']].copy()

log(f"\nSurvivors:")
for _, r in survivors.iterrows():
    log(f"  ✓ {r['window']:15s} × {r['signal']:25s} (β2={r['phase_coef_m2']:+.4f}, p={r['phase_p_m2']:.4f})")

log(f"\nConfounded:")
for _, r in confounded.iterrows():
    log(f"  ✗ {r['window']:15s} × {r['signal']:25s} (β2={r['phase_coef_m2']:+.4f}, p={r['phase_p_m2']:.4f})")

if len(survivors) > 0:
    survivors['abs_delta'] = survivors['delta_phase'].abs()
    best = survivors.loc[survivors['abs_delta'].idxmin()]
    log(f"\nCleanest dissociation (smallest Δ): {best['window']} × {best['signal']}")
    log(f"  Phase effect: β1={best['phase_coef_m1']:+.4f} → β2={best['phase_coef_m2']:+.4f} (Δ={best['delta_phase']:+.4f})")
    
    biggest_d = survivors.loc[survivors['phase_cohens_d_m1'].abs().idxmax()]
    log(f"\nStrongest surviving effect: {biggest_d['window']} × {biggest_d['signal']} (d={biggest_d['phase_cohens_d_m1']:.3f})")

# ═══════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════

log("\n\nGenerating visualisation...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes_flat = axes.flatten()

focus_signals = ['reorientation_rate', 'pressing_accuracy']
window_labels = {
    '10min_rolling': '10-min Rolling',
    '15min_decaying': '15-min Exponential Decay',
    'half_game': 'Half-Game Mean',
    'full_game': 'Full-Game Mean',
}

for idx, wtype in enumerate(WINDOW_TYPES):
    ax = axes_flat[idx]
    phys_col = f'phys_{wtype}'
    
    for si, sig in enumerate(focus_signals):
        sig_col = f'{sig}_{wtype}'
        sub = df[[sig_col, phys_col, 'phase']].dropna().copy()
        if len(sub) < 50:
            continue
        
        sub['phase01'] = (sub['phase'] - 1).astype(float)
        sub['phys_scaled'] = (sub[phys_col] - sub[phys_col].mean()) / sub[phys_col].std()
        
        m1 = smf.ols(f'{sig_col} ~ phase01', data=sub).fit()
        m2 = smf.ols(f'{sig_col} ~ phase01 + phys_scaled', data=sub).fit()
        
        beta1 = m1.params['phase01']
        beta2 = m2.params['phase01']
        
        means = sub.groupby('phase')[sig_col].mean()
        sems = sub.groupby('phase')[sig_col].sem()
        surv = m2.pvalues['phase01'] < 0.05
        
        offset = si * 0.15
        x_pos = [1 + offset, 2 + offset]
        
        color = '#2196F3' if si == 0 else '#FF9800'
        alpha_surv = 1.0 if surv else 0.35
        label = f"{sig.replace('_', ' ').title()}{' (✓)' if surv else ' (✗)'}"
        
        ax.errorbar(x_pos, means.values, yerr=sems.values * 1.96,
                     fmt='o-', color=color, capsize=4, capthick=1.5,
                     markersize=8, linewidth=2, alpha=alpha_surv, label=label)
    
    ax.set_title(window_labels[wtype], fontsize=13, fontweight='bold')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Phase 1', 'Phase 2'])
    ax.set_ylabel('Cognitive Signal Value', fontsize=10)
    ax.legend(fontsize=9, loc='best')
    ax.axvline(1.5, color='gray', linestyle=':', alpha=0.3)
    ax.grid(True, alpha=0.3)

# Summary panel in bottom-right
ax = axes_flat[3]
ax.axis('off')
summary_lines = ["DISSOCIATION SUMMARY", "="*30, ""]
for wtype in WINDOW_TYPES:
    r = results_df[results_df['window'] == wtype]
    summary_lines.append(f"{window_labels[wtype]}:")
    for _, row in r.iterrows():
        sig_short = row['signal'].replace('_', ' ').title()
        surv = "✓" if row['survives_control'] else "✗"
        summary_lines.append(f"  {sig_short:25s} {surv}  "
                            f"β1={row['phase_coef_m1']:+.3f}→β2={row['phase_coef_m2']:+.3f}")
    summary_lines.append("")

ax.text(0.02, 0.98, '\n'.join(summary_lines), transform=ax.transAxes,
        fontsize=7, fontfamily='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.suptitle('Rolling Window Fatigue Analysis: Proper Dissociation',
             fontsize=15, fontweight='bold', y=1.01)

plt.savefig('focus-fatigue/outputs/analysis/rolling_window_figure.png',
            dpi=200, bbox_inches='tight')
log("Figure saved.")

# ── Save results table as markdown ──
md_lines = []
md_lines.append("# Rolling Window Fatigue Analysis: Proper Dissociation")
md_lines.append("")
md_lines.append("## Methodology")
md_lines.append("")
md_lines.append("For each (player, game), we compute four window types for BOTH physical load and cognitive signals:")
md_lines.append("")
md_lines.append("| Window | Description |")
md_lines.append("|--------|-------------|")
md_lines.append("| 10-min Rolling (lagging) | Mean of current + previous 5-min block |")
md_lines.append("| 15-min Exponential Decay | Exponentially weighted mean of all preceding blocks (τ=15 min) |")
md_lines.append("| Half-Game | Mean across all blocks within the same phase (half) |")
md_lines.append("| Full-Game | Mean across all blocks in the game |")
md_lines.append("")
md_lines.append("For each (window × cognitive signal) combination, three models are fit:")
md_lines.append("")
md_lines.append("1. **M1 (Univariate):** `signal_windowed ~ phase`")
md_lines.append("2. **M2 (Controlled):** `signal_windowed ~ phase + phys_load_windowed`")
md_lines.append("3. **M3 (Interaction):** `signal_windowed ~ phase * phys_load_windowed`")
md_lines.append("")
md_lines.append("The key metric is **Δ** = β₂(phase) − β₁(phase): how much the phase effect attenuates when controlling for physical load on the same window. A signal **survives** if the phase coefficient remains significant (p<0.05) in M2.")
md_lines.append("")
md_lines.append("## Results Table")
md_lines.append("")
md_lines.append("| Window | Signal | N | M1 β(phase) | M1 p | Cohen's d | M2 β(phase) | M2 p | Δ | Attn% | Survives? | M2 β(phys) | M2 p(phys) |")
md_lines.append("|--------|--------|---|-------------|------|-----------|-------------|------|----|-------|-----------|------------|------------|")

for _, r in results_df.iterrows():
    surv = "✓" if r['survives_control'] else "✗"
    d_str = f"{r['phase_cohens_d_m1']:.3f}" if not np.isnan(r['phase_cohens_d_m1']) else "N/A"
    attn_str = f"{r['pct_attn']:.0f}%" if not np.isnan(r['pct_attn']) else "N/A"
    md_lines.append(
        f"| {r['window']} | {r['signal']} | {r['n_obs']} | "
        f"{r['phase_coef_m1']:.4f} | {r['phase_p_m1']:.4f} | {d_str} | "
        f"{r['phase_coef_m2']:.4f} | {r['phase_p_m2']:.4f} | "
        f"{r['delta_phase']:.4f} | {attn_str} | {surv} | "
        f"{r['phys_coef_m2']:.4f} | {r['phys_p_m2']:.4f} |"
    )

md_lines.append("")
md_lines.append("## Key Findings")
md_lines.append("")

survivors_sorted = survivors.sort_values('abs_delta')
md_lines.append(f"**Signals that survive (phase still significant after controlling for physical load):**")
for _, r in survivors_sorted.iterrows():
    md_lines.append(f"- **{r['signal']}** on {r['window']} window: β₁={r['phase_coef_m1']:.4f} → β₂={r['phase_coef_m2']:.4f} (Δ={r['delta_phase']:.4f}, d={r['phase_cohens_d_m1']:.3f})")

md_lines.append("")
md_lines.append(f"**Signals that get confounded (phase no longer significant after controlling for physical load):**")
for _, r in confounded.iterrows():
    md_lines.append(f"- **{r['signal']}** on {r['window']} window: β₁={r['phase_coef_m1']:.4f} → β₂={r['phase_coef_m2']:.4f} (p={r['phase_p_m2']:.4f})")

if len(survivors) > 0:
    best_row = survivors_sorted.iloc[0]
    md_lines.append("")
    md_lines.append(f"### Cleanest Dissociation")
    md_lines.append(f"**{best_row['window']} × {best_row['signal']}** shows the smallest Δ ({best_row['delta_phase']:.4f}), "
                   f"meaning the phase effect is nearly unchanged when controlling for physical load. "
                   f"This is the strongest evidence for a genuinely cognitive fatigue signal.")

with open('focus-fatigue/outputs/analysis/rolling_window_results.md', 'w') as f:
    f.write('\n'.join(md_lines))
log("Markdown report saved.")

log("\n\nDone! ✅")
