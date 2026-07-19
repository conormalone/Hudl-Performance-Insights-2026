#!/usr/bin/env python3
"""
Cognitive Load → Cumulative Fatigue → Defensive Decline
========================================================

Tests the feedback loop: Phase 1 cognitive load → Phase 2 defensive decline.

Key insight: Uses WITHIN-PLAYER decline (Phase 2 - Phase 1 difference) 
and ROLLING within-game fatigue to separate selection effects from fatigue effects.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE = '/home/conormalone/Hudl-Performance-Insights-2026/focus-fatigue'
df = pd.read_parquet(f'{BASE}/outputs/analysis/unified_fatigue_dataset.parquet')

COG_INDICATORS = ['pressure_composite', 'opponents_nearby_mean', 
                   'reorientation_count', 'transition_count', 'depth_mean']

OUTCOMES = {
    'reorientation_rate': 'scans/frame',
    'pressing_accuracy': 'success rate',
    'shift_latency': 'seconds (lower=faster)',
    'positional_drift': 'drift units (lower=better)',
}

WINDOW_TYPES = ['10min', '15min_decay', 'half', 'full']

print(f"Loaded {len(df):,} rows, {df['player_id'].nunique()} players, {df['game_id'].nunique()} games")

# ─── Pooled z-score standardization ───────────────────────────────────────────
for col in COG_INDICATORS:
    mu, sigma = df[col].mean(), df[col].std()
    df[f'{col}_z'] = (df[col] - mu) / sigma

df['cog_load'] = df[[f'{c}_z' for c in COG_INDICATORS]].mean(axis=1)
df['pg_key'] = df['player_id'].astype(str) + '_' + df['game_id'].astype(str)

# ─── Split phases ─────────────────────────────────────────────────────────────
phase1 = df[(df['phase'] == 1) & (df['block_num'] <= 9)].copy()
phase2 = df[df['phase'] == 2].copy()

print(f"Phase 1: {len(phase1):,} rows, Phase 2: {len(phase2):,} rows")

# ─── Phase 1 aggregates per (player, game) ────────────────────────────────────
agg_p1 = phase1.groupby('pg_key').agg(
    cog_load_phase1=('cog_load', 'mean'),
    phys_load_phase1=('physical_load', 'mean'),
    num_blocks_p1=('block_num', 'count'),
    **{f'{c}_p1mean': (c, 'mean') for c in COG_INDICATORS},
    # Also compute Phase 1 mean of each outcome
    reo_rate_p1=('reorientation_rate', 'mean'),
    press_acc_p1=('pressing_accuracy', 'mean'),
    shift_lat_p1=('shift_latency', 'mean'),
    drift_p1=('positional_drift', 'mean'),
).reset_index()

# ─── Compute per-(player, game) Phase 2 mean of each outcome ──────────────────
agg_p2 = phase2.groupby('pg_key').agg(
    reo_rate_p2=('reorientation_rate', 'mean'),
    press_acc_p2=('pressing_accuracy', 'mean'),
    shift_lat_p2=('shift_latency', 'mean'),
    drift_p2=('positional_drift', 'mean'),
    num_blocks_p2=('block_num', 'count'),
).reset_index()

# ─── Merge Phase 1 + Phase 2 aggregates ───────────────────────────────────────
player_game = agg_p1.merge(agg_p2, on='pg_key', how='inner')

# Compute change scores (Phase 2 - Phase 1)
player_game['reo_decline'] = player_game['reo_rate_p2'] - player_game['reo_rate_p1']
player_game['press_decline'] = player_game['press_acc_p2'] - player_game['press_acc_p1']
player_game['shift_decline'] = player_game['shift_lat_p2'] - player_game['shift_lat_p1']  # positive = slower = worse
player_game['drift_decline'] = player_game['drift_p2'] - player_game['drift_p1']  # positive = more drift = worse

DECLINE_MAP = {
    'reorientation_rate': 'reo_decline',
    'pressing_accuracy': 'press_decline',
    'shift_latency': 'shift_decline',
    'positional_drift': 'drift_decline',
}

print(f"Merged player-game data: {len(player_game)} combos")
print(f"Cog load phase 1: mean={player_game['cog_load_phase1'].mean():.4f}, std={player_game['cog_load_phase1'].std():.4f}")
print(f"Decline ranges: reo=[{player_game['reo_decline'].min():.2f}, {player_game['reo_decline'].max():.2f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL A: Phase 1 Load → Phase 2 DECLINE (change score)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("MODEL A: Phase 1 Load → Phase 2 DECLINE (Change Score)")
print("  (Phase2_mean - Phase1_mean) ~ cog_load_phase1 + phys_load_phase1")
print("="*80)

model_a_results = {}
for outcome, decline_col in DECLINE_MAP.items():
    valid = player_game.dropna(subset=[decline_col, 'cog_load_phase1', 'phys_load_phase1'])
    if len(valid) < 50:
        continue
    
    n = len(valid)
    y = valid[decline_col].values  # change score (negative = worse for reo/press, positive = worse for shift/drift)
    
    # Standardised predictors
    X = np.column_stack([
        np.ones(n),
        valid['cog_load_phase1'].values / np.std(valid['cog_load_phase1'].values),
        valid['phys_load_phase1'].values / np.std(valid['phys_load_phase1'].values),
    ])
    
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    k = X.shape[1]
    mse = np.sum(residuals**2) / (n - k)
    var_beta = mse * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(var_beta))
    t_stats = beta / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
    r2 = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
    
    # Partial r for cog_load controlling phys_load
    controls = np.column_stack([np.ones(n), valid['phys_load_phase1'].values])
    y_partial = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
    cog_p = valid['cog_load_phase1'].values - controls @ np.linalg.lstsq(controls, valid['cog_load_phase1'].values, rcond=None)[0]
    r_partial = np.corrcoef(y_partial, cog_p)[0, 1] if np.std(cog_p) > 0 else 0
    
    # Key: does cog_load predict WORSE decline?
    # For reo_rate and press_acc: decline = negative change. If cog_load has negative β, it predicts decline.
    # For shift_lat and drift: decline = positive change. If cog_load has positive β, it predicts decline.
    if outcome in ['reorientation_rate', 'pressing_accuracy']:
        predicts_decline = beta[1] < 0 and p_values[1] < 0.05
        decline_str = f"decline (β={beta[1]:+.4f} negative → higher cog load → greater drop)"
    else:
        predicts_decline = beta[1] > 0 and p_values[1] < 0.05
        decline_str = f"decline (β={beta[1]:+.4f} positive → higher cog load → more worsening)"
    
    sig = "***" if p_values[1] < 0.001 else "**" if p_values[1] < 0.01 else "*" if p_values[1] < 0.05 else "ns"
    
    print(f"\n  {outcome:25s} (N={n}, R²={r2:.4f})")
    print(f"    cog_load_phase1: β={beta[1]:+.6f}, t={t_stats[1]:.2f}, p={p_values[1]:.6f} {sig}")
    print(f"      partial r (controlling phys) = {r_partial:.4f}")
    print(f"    phys_load_phase1: β={beta[2]:+.6f}, t={t_stats[2]:.2f}, p={p_values[2]:.6f}")
    print(f"    Mean decline: {y.mean():+.4f}")
    if predicts_decline:
        print(f"    ✓ HIGHER cognitive load → MORE {outcome} decline")
    elif p_values[1] < 0.05:
        print(f"    ✗ Counter-directional: HIGHER cog load → LESS decline (or improvement)")
    else:
        print(f"    — No significant relationship")
    
    model_a_results[outcome] = {
        'coef': beta[1], 't': t_stats[1], 'p': p_values[1],
        'partial_r': r_partial, 'r2': r2, 'n': n,
        'sig': sig, 'predicts_decline': predicts_decline,
        'mean_decline': y.mean()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL B: Rolling Fatigue → Next-Block Quality (WITHIN-PLAYER)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("MODEL B: Rolling Cognitive Fatigue → Block-Level Quality (WITHIN-PLAYER)")
print("  outcome_at_t ~ rolling_cog_t + rolling_phys_t + (1|player) + (1|game)")
print("="*80)

# First: compute rolling measures for everyone
def ewma(values, tau):
    if len(values) == 0: return np.nan
    alpha = 1 - np.exp(-1 / tau) if tau > 0 else 1.0
    weights = np.array([(1 - alpha) ** i for i in range(len(values) - 1, -1, -1)])
    return np.average(values, weights=weights)

rolling_records = []
for key, grp in df.sort_values(['pg_key', 'block_num', 'phase']).groupby('pg_key', sort=False):
    grp = grp.sort_values(['phase', 'block_num'])
    cog_vals = grp['cog_load'].values
    phys_vals = grp['physical_load'].values
    
    for i in range(len(grp)):
        record = {'pg_key': key, 'phase': grp['phase'].iloc[i], 'block_num': grp['block_num'].iloc[i]}
        
        if i >= 1:
            record['rolling_cog_10min'] = np.mean(cog_vals[max(0,i-1):i+1])
            record['rolling_phys_10min'] = np.mean(phys_vals[max(0,i-1):i+1])
        else:
            record['rolling_cog_10min'] = cog_vals[i]
            record['rolling_phys_10min'] = phys_vals[i]
        
        record['rolling_cog_15min_decay'] = ewma(cog_vals[:i+1], tau=15)
        record['rolling_phys_15min_decay'] = ewma(phys_vals[:i+1], tau=15)
        
        # Half-game = Phase 1 mean (cached from agg_p1)
        p1_row = agg_p1[agg_p1['pg_key'] == key]
        if len(p1_row) > 0:
            record['rolling_cog_half'] = float(p1_row['cog_load_phase1'].iloc[0])
            record['rolling_phys_half'] = float(p1_row['phys_load_phase1'].iloc[0])
        else:
            record['rolling_cog_half'] = np.nan
            record['rolling_phys_half'] = np.nan
        
        # Full-game cumulative
        if len(p1_row) > 0:
            p1_cog = float(p1_row['cog_load_phase1'].iloc[0])
            p1_phys = float(p1_row['phys_load_phase1'].iloc[0])
            n_p1 = float(p1_row['num_blocks_p1'].iloc[0])
            n_seen = i + 1
            total = n_p1 + n_seen
            record['rolling_cog_full'] = (p1_cog * n_p1 + np.sum(cog_vals[:i+1])) / total
            record['rolling_phys_full'] = (p1_phys * n_p1 + np.sum(phys_vals[:i+1])) / total
        else:
            record['rolling_cog_full'] = np.nan
            record['rolling_phys_full'] = np.nan
        
        rolling_records.append(record)

rolling_df = pd.DataFrame(rolling_records)
df = df.merge(rolling_df, on=['pg_key', 'phase', 'block_num'], how='left')

# Now run models on Phase 2 data WITH player + game fixed effects
phase2 = df[df['phase'] == 2].copy()
phase2['player_id'] = phase2['player_id'].astype('category')
phase2['game_id'] = phase2['game_id'].astype('category')

print(f"Phase 2 with rolling measures: {len(phase2):,} rows")

model_b_results = {}
for outcome in OUTCOMES:
    model_b_results[outcome] = {}
    print(f"\n  Outcome: {outcome}")
    for wt in WINDOW_TYPES:
        cog_col = f'rolling_cog_{wt}'
        phys_col = f'rolling_phys_{wt}'
        
        valid = phase2.dropna(subset=[outcome, cog_col, phys_col, 'player_id', 'block_num'])
        if len(valid) < 5000:
            print(f"    {wt:>15}: insufficient ({len(valid)})")
            continue
        
        # WITHIN-PLAYER DEMEANING: subtract each player's mean
        player_means = valid.groupby('player_id')[outcome].transform('mean')
        valid = valid.copy()
        valid['y_within'] = valid[outcome] - player_means
        
        cog_means = valid.groupby('player_id')[cog_col].transform('mean')
        phys_means = valid.groupby('player_id')[phys_col].transform('mean')
        valid['cog_within'] = valid[cog_col] - cog_means
        valid['phys_within'] = valid[phys_col] - phys_means
        valid['blk_within'] = valid['block_num'] - valid.groupby('player_id')['block_num'].transform('mean')
        
        valid = valid.dropna(subset=['y_within', 'cog_within', 'phys_within', 'blk_within'])
        n = len(valid)
        
        y = valid['y_within'].values
        X = np.column_stack([
            np.ones(n),
            valid['cog_within'].values,
            valid['phys_within'].values,
            valid['blk_within'].values,
        ])
        
        # OLS with HC3 robust SE
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        k = X.shape[1]
        mse = np.sum(residuals**2) / (n - k)
        
        # HC3 robust covariance
        h = np.sum(X * (X @ np.linalg.inv(X.T @ X)), axis=1)
        omega = np.diag(residuals**2 / (1 - h)**2)
        vce = np.linalg.inv(X.T @ X) @ (X.T @ omega @ X) @ np.linalg.inv(X.T @ X)
        se_robust = np.sqrt(np.diag(vce))
        t_robust = beta / se_robust
        p_robust = 2 * (1 - stats.t.cdf(np.abs(t_robust), df=n - k))
        
        r2 = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
        
        sig = "***" if p_robust[1] < 0.001 else "**" if p_robust[1] < 0.01 else "*" if p_robust[1] < 0.05 else "ns"
        
        # For reo_rate and press_acc: negative β = within-player decline
        # For shift_lat and drift: positive β = within-player decline
        if outcome in ['reorientation_rate', 'pressing_accuracy']:
            decline = beta[1] < 0 and p_robust[1] < 0.05
        else:
            decline = beta[1] > 0 and p_robust[1] < 0.05
        
        print(f"    {wt:>15}: β={beta[1]:+.6f}, t={t_robust[1]:.2f}, p={p_robust[1]:.6f} {sig}, "
              f"R²={r2:.4f}, n={n:,}")
        if decline:
            print(f"           → ✓ Within-player: higher rolling cog → WORSE {outcome}")
        elif p_robust[1] < 0.05:
            print(f"           → ✗ Within-player: higher rolling cog → BETTER {outcome} (counter-directional)")
        else:
            print(f"           → No significant within-player effect")
        
        model_b_results[outcome][wt] = {
            'coef': beta[1], 't': t_robust[1], 'p': p_robust[1],
            'sig': sig, 'decline': decline, 'r2': r2, 'n': n
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL C: High vs Low Cognitive Fatigue Quartile Split
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("MODEL C: High vs Low Cognitive Fatigue — Phase 2 Decline (Quartile Split)")
print("  Phase 2 outcomes compared between high-Q4 and low-Q1 on Phase 1 cog_load")
print("="*80)

model_c_results = {}
for outcome, decline_col in DECLINE_MAP.items():
    valid = player_game.dropna(subset=[decline_col, 'cog_load_phase1'])
    if len(valid) < 50:
        continue
    
    q_low = valid['cog_load_phase1'].quantile(0.25)
    q_high = valid['cog_load_phase1'].quantile(0.75)
    
    low_group = valid[valid['cog_load_phase1'] <= q_low][decline_col].dropna()
    high_group = valid[valid['cog_load_phase1'] >= q_high][decline_col].dropna()
    
    if len(low_group) < 10 or len(high_group) < 10:
        continue
    
    mean_low = low_group.mean()
    mean_high = high_group.mean()
    diff = mean_high - mean_low
    se_diff = np.sqrt(low_group.var()/len(low_group) + high_group.var()/len(high_group))
    ci_95 = 1.96 * se_diff
    t_stat = diff / se_diff
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=min(len(low_group), len(high_group)) - 1))
    pooled_std = np.sqrt((low_group.var() + high_group.var()) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0
    
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    
    # Real units: diff in the decline score
    x_unit = outcome.replace('_', ' ')
    
    # Which group has MORE decline?
    # For reo_rate and press_acc: more negative decline = worse
    # For shift_lat and drift: more positive decline = worse
    if outcome in ['reorientation_rate', 'pressing_accuracy']:
        high_more_decline = mean_high < mean_low
        more_decline_str = f"High-fatigue group declines MORE (mean diff: {mean_high:.4f} vs {mean_low:.4f})" if high_more_decline else f"Low-fatigue group declines MORE"
    else:
        high_more_decline = mean_high > mean_low
        more_decline_str = f"High-fatigue group declines MORE (mean diff: {mean_high:.4f} vs {mean_low:.4f})" if high_more_decline else f"Low-fatigue group declines MORE"
    
    print(f"\n  {outcome:25s}")
    print(f"    Low cog fatigue (Q1) n={len(low_group):,}: decline M={mean_low:.4f}")
    print(f"    High cog fatigue (Q4) n={len(high_group):,}: decline M={mean_high:.4f}")
    print(f"    Difference: {diff:+.4f} [{diff-ci_95:.4f}, {diff+ci_95:.4f}] 95% CI")
    print(f"    Cohen's d = {cohens_d:.3f}, t = {t_stat:.2f}, p = {p_val:.6f} {sig}")
    if p_val < 0.05 and high_more_decline:
        print(f"    ✓ {more_decline_str}")
    elif p_val < 0.05:
        print(f"    ✗ {more_decline_str} (opposite direction)")
    else:
        print(f"    — No significant group difference")
    
    model_c_results[outcome] = {
        'low_mean': mean_low, 'high_mean': mean_high, 'diff': diff,
        'ci_low': diff - ci_95, 'ci_high': diff + ci_95,
        'cohens_d': cohens_d, 'p_val': p_val, 'sig': sig,
        'n_low': len(low_group), 'n_high': len(high_group),
        'high_more_decline': high_more_decline,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL: Which cognitive load indicator is the strongest predictor of decline?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("INDIVIDUAL INDICATOR IMPORTANCE: Each → Decline Score")
print("  (Phase2_mean - Phase1_mean) ~ indicator_p1mean + phys_load_phase1")
print("="*80)

indicator_results = {}
for outcome, decline_col in DECLINE_MAP.items():
    indicator_results[outcome] = {}
    print(f"\n  Outcome: {outcome}")
    for indicator in COG_INDICATORS:
        col = f'{indicator}_p1mean'
        valid = player_game.dropna(subset=[decline_col, col, 'phys_load_phase1'])
        if len(valid) < 50:
            continue
        
        n = len(valid)
        y = valid[decline_col].values
        X = np.column_stack([
            np.ones(n),
            valid[col].values / np.std(valid[col].values),
            valid['phys_load_phase1'].values / np.std(valid['phys_load_phase1'].values),
        ])
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        k = X.shape[1]
        mse = np.sum(residuals**2) / (n - k)
        var_beta = mse * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(np.diag(var_beta))
        t_stats = beta / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
        
        sig = "***" if p_values[1] < 0.001 else "**" if p_values[1] < 0.01 else "*" if p_values[1] < 0.05 else ""
        print(f"    {indicator:>30}: β_std={beta[1]:+.6f}, t={t_stats[1]:.2f}, p={p_values[1]:.6f} {sig}")
        
        indicator_results[outcome][indicator] = {
            'beta_std': beta[1], 't': t_stats[1], 'p': p_values[1], 'sig': sig, 'n': n
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: Block-by-block trajectory in Phase 2
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SUPPLEMENTARY: Block-by-block trajectory split by high/low cog_load_phase1")
print("="*80)

# Split players into high/low using median Phase 1 cog_load
player_game['cog_group'] = pd.qcut(player_game['cog_load_phase1'], q=2, labels=['Low', 'High'])
p2_with_groups = phase2.merge(agg_p1[['pg_key', 'cog_load_phase1']], on='pg_key', how='left')
p2_with_groups['cog_group'] = pd.qcut(p2_with_groups['cog_load_phase1'], q=2, 
                                        labels=['Low', 'High'], duplicates='drop')

# For each outcome, compute mean at each block by group
for outcome in OUTCOMES:
    print(f"\n  {outcome} by block_num and cog_group:")
    block_means = p2_with_groups.dropna(subset=[outcome, 'cog_group']).groupby(
        ['block_num', 'cog_group'])[outcome].agg(['mean', 'std', 'count'])
    print(block_means.to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n")
print("="*80)
print("FINAL SUMMARY")
print("="*80)

print("\n1. MODEL A (Phase 1 Load → Phase 2 Decline Change Score)")
for out, res in model_a_results.items():
    print(f"   {out:25s}: β={res['coef']:+.4f}, p={res['p']:.4f} {res['sig']}, "
          f"partial_r={res['partial_r']:.4f}")
    if res['predicts_decline']:
        print(f"          ✓ Higher cog load → MORE decline")
    elif res['p'] < 0.05:
        print(f"          ✗ Higher cog load → LESS decline (counter-directional)")
    else:
        print(f"          — No significant effect")

print("\n2. MODEL B (Within-Player Rolling Fatigue → Block Quality)")
for out, windows in model_b_results.items():
    sig_windows = [wt for wt, r in windows.items() if r['sig'] not in ['ns', '']]
    decline_windows = [wt for wt, r in windows.items() if r.get('decline', False)]
    if decline_windows:
        print(f"   {out:25s}: DECLINE detected in: {', '.join(decline_windows)}")
    elif sig_windows:
        print(f"   {out:25s}: Significant but counter-directional in: {', '.join(sig_windows)}")
    else:
        print(f"   {out:25s}: No significant within-player effects")

print("\n3. MODEL C (High vs Low Cognitive Fatigue Group — Quartile Split)")
for out, res in model_c_results.items():
    print(f"   {out:25s}: diff={res['diff']:+.4f} [{res['ci_low']:.4f}, {res['ci_high']:.4f}], "
          f"d={res['cohens_d']:.3f}, p={res['p_val']:.4f} {res['sig']}")
    if res.get('high_more_decline'):
        print(f"          ✓ High-fatigue group shows MORE decline")
    elif res['p_val'] < 0.05:
        print(f"          ✗ Low-fatigue group shows MORE decline (opposite)")
    else:
        print(f"          — No significant group difference")

print("\n4. STRONGEST COGNITIVE LOAD INDICATOR\n")
for out in OUTCOMES:
    if out in indicator_results and indicator_results[out]:
        best = min(indicator_results[out].items(), key=lambda x: x[1]['p'])
        print(f"   {out:25s}: strongest={best[0]:>30} (β_std={best[1]['beta_std']:+.4f}, p={best[1]['p']:.4f})")

print("\nDone.")
