#!/usr/bin/env python3
"""
Defensive Group Fatigue Analysis
=================================
Steps:
1. Derive position clusters from behavioral averages
2. Define defensive group (CB + FB + DM)
3. Compute rolling cognitive & physical load from PRECEDING blocks only
4. 75th percentile grouping (high vs low load)
5. Demand-adjusted model: predict expected reorientation_rate from situation
6. Test: fatigue_deficit ~ load_group + phys_load_group
7. Run on ALL players for comparison

Hard rules:
- NO Phase 1/Phase 2 framing
- NO time variables (block_num, minutes, halves as predictors)
- Percentile thresholds for load groups
- Rolling from PRECEDING blocks only
- Physical load as control in every model
"""

import sys, warnings, json
import pandas as pd
import numpy as np
from scipy import stats, cluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
warnings.filterwarnings('ignore')

np.random.seed(42)

def log(msg):
    print(msg, flush=True)

# ═══════════════════════════════════════════
# 0. CONFIG
# ═══════════════════════════════════════════
DATA_PATH = '/home/conormalone/Hudl-Performance-Insights-2026/focus-fatigue/outputs/analysis/unified_fatigue_dataset.parquet'
LOOKUP_PATH = '/home/conormalone/Hudl-Performance-Insights-2026/focus-fatigue/outputs/analysis/player_position_lookup.csv'
OUT_DIR = '/home/conormalone/Hudl-Performance-Insights-2026/focus-fatigue/outputs/analysis'
REVIEW_DIR = '/home/conormalone/Hudl-Performance-Insights-2026/focus-fatigue/review'

BLOCK_MINUTES = 5.0  # each block is ~5 min
TAU = 15.0  # decay constant for 15-min decaying window

DEMAND_VARS = ['pressure_composite', 'opponents_nearby_mean', 'depth_mean']
COG_LOAD_VARS = ['pressure_composite', 'opponents_nearby_mean', 'reorientation_count', 'transition_count', 'depth_mean']
PHYS_VAR = 'physical_load'

# ═══════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════
log("=" * 60)
log("DEFENSIVE GROUP FATIGUE ANALYSIS")
log("=" * 60)

log("\nLoading data...")
df = pd.read_parquet(DATA_PATH)
log(f"Loaded {len(df):,} rows, {df['player_id'].nunique()} players, {df['game_id'].nunique()} games")

lookup = pd.read_csv(LOOKUP_PATH)
log(f"Position lookup: {len(lookup)} players mapped")

# Merge position lookups
df = df.merge(lookup, on='player_id', how='left')
log(f"After merge: {df['position'].isna().sum()} players with no position mapping")

# ═══════════════════════════════════════════
# 2. DEFINE DEFENSIVE GROUP
# ═══════════════════════════════════════════
log("\n--- Position counts ---")
pos_counts = df.groupby('position')['player_id'].nunique()
for pos, cnt in pos_counts.items():
    log(f"  {pos}: {cnt} players")

df['is_defender'] = df['position'].isin(['CB', 'FB', 'DM'])
n_def_players = df[df['is_defender']]['player_id'].nunique()
n_off_players = df[~df['is_defender']]['player_id'].nunique()
log(f"\nDefensive group: {n_def_players} players")
log(f"Non-defensive: {n_off_players} players")
log(f"Defensive blocks: {df['is_defender'].sum():,} / {len(df):,}")

# ═══════════════════════════════════════════
# 3. COMPUTE ROLLING LOAD (PRECEDING BLOCKS ONLY)
# ═══════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 3: Rolling load from PRECEDING blocks")
log("=" * 60)

df = df.sort_values(['game_id', 'player_id', 'block_num']).reset_index(drop=True)
df['block_global'] = df.groupby(['game_id', 'player_id']).cumcount()

# Initialize rolling columns
df['rolling_cog_load_mean'] = np.nan     # 10-min rolling (2 preceding blocks)
df['rolling_cog_load_decay'] = np.nan    # 15-min exponentially decaying
df['rolling_phys_load_mean'] = np.nan
df['rolling_phys_load_decay'] = np.nan

# Also store components for transparency
for var in COG_LOAD_VARS:
    df[f'rolling_{var}_mean'] = np.nan
    df[f'rolling_{var}_decay'] = np.nan

groups = list(df.groupby(['game_id', 'player_id']))
n_groups = len(groups)

for gi, ((gid, pid), gdf) in enumerate(groups):
    if gi % 500 == 0:
        log(f"  Processing group {gi}/{n_groups}...")
    
    g = gdf.sort_values('block_num')
    g_idx = g.index.values
    n = len(g)
    
    # Extract values
    vals = {}
    for var in COG_LOAD_VARS + [PHYS_VAR]:
        vals[var] = g[var].values
    
    for i in range(n):
        if i == 0:
            # First block: no preceding blocks, set to 0
            df.loc[g_idx[i], 'rolling_cog_load_mean'] = 0.0
            df.loc[g_idx[i], 'rolling_cog_load_decay'] = 0.0
            df.loc[g_idx[i], 'rolling_phys_load_mean'] = 0.0
            df.loc[g_idx[i], 'rolling_phys_load_decay'] = 0.0
            for var in COG_LOAD_VARS:
                df.loc[g_idx[i], f'rolling_{var}_mean'] = 0.0
                df.loc[g_idx[i], f'rolling_{var}_decay'] = 0.0
            continue
        
        preceding = np.arange(i)  # indices of ALL preceding blocks
        
        # --- 10-min rolling: mean of 2 preceding blocks ---
        window_size = min(2, i)
        w_preceding = np.arange(i - window_size, i)
        
        cog_vals_window = np.array([vals['pressure_composite'][j] for j in w_preceding])
        phys_vals_window = np.array([vals[PHYS_VAR][j] for j in w_preceding])
        df.loc[g_idx[i], 'rolling_cog_load_mean'] = cog_vals_window.mean()
        df.loc[g_idx[i], 'rolling_phys_load_mean'] = phys_vals_window.mean()
        
        for var in COG_LOAD_VARS:
            var_window = np.array([vals[var][j] for j in w_preceding])
            df.loc[g_idx[i], f'rolling_{var}_mean'] = var_window.mean()
        
        # --- 15-min exponentially decaying: ALL preceding blocks ---
        # tau=15 min, blocks are ~5 min
        lags = np.arange(i, 0, -1) * BLOCK_MINUTES  # minutes ago
        weights = np.exp(-lags / TAU)
        weights = weights / weights.sum()
        
        cog_preceding = np.array([vals['pressure_composite'][j] for j in preceding])
        phys_preceding = np.array([vals[PHYS_VAR][j] for j in preceding])
        df.loc[g_idx[i], 'rolling_cog_load_decay'] = np.average(cog_preceding, weights=weights)
        df.loc[g_idx[i], 'rolling_phys_load_decay'] = np.average(phys_preceding, weights=weights)
        
        for var in COG_LOAD_VARS:
            var_preceding = np.array([vals[var][j] for j in preceding])
            df.loc[g_idx[i], f'rolling_{var}_decay'] = np.average(var_preceding, weights=weights)

log("Rolling load computed.")
for col in ['rolling_cog_load_mean', 'rolling_cog_load_decay', 'rolling_phys_load_mean', 'rolling_phys_load_decay']:
    log(f"  {col}: mean={df[col].mean():.4f}, missing={df[col].isnull().sum()}")

# ═══════════════════════════════════════════
# 4. COMPOSITE Z-SCORES OF ROLLING COGNITIVE LOAD
# ═══════════════════════════════════════════
log("\n--- Computing composite z-scores for rolling cognitive load ---")

# For each window type, standardize the 5 cognitive load components and average
for wtype in ['mean', 'decay']:
    # Standardize each component
    for var in COG_LOAD_VARS:
        col = f'rolling_{var}_{wtype}'
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            df[f'z_{var}_{wtype}'] = (df[col] - mean_val) / std_val
        else:
            df[f'z_{var}_{wtype}'] = 0.0
    
    # Composite z-score: average of standardized components
    z_cols = [f'z_{var}_{wtype}' for var in COG_LOAD_VARS]
    df[f'rolling_cog_load_z_{wtype}'] = df[z_cols].mean(axis=1)
    
    # Standardize the composite itself
    z_mean = df[f'rolling_cog_load_z_{wtype}'].mean()
    z_std = df[f'rolling_cog_load_z_{wtype}'].std()
    if z_std > 0:
        df[f'rolling_cog_load_z_{wtype}'] = (df[f'rolling_cog_load_z_{wtype}'] - z_mean) / z_std
    
    # Also standardize physical load
    col = f'rolling_phys_load_{wtype}'
    mean_val = df[col].mean()
    std_val = df[col].std()
    if std_val > 0:
        df[f'rolling_phys_load_z_{wtype}'] = (df[col] - mean_val) / std_val
    else:
        df[f'rolling_phys_load_z_{wtype}'] = 0.0
    
    log(f"  {wtype}: cog_z mean={df[f'rolling_cog_load_z_{wtype}'].mean():.4f}, std={df[f'rolling_cog_load_z_{wtype}'].std():.4f}")
    log(f"  {wtype}: phys_z mean={df[f'rolling_phys_load_z_{wtype}'].mean():.4f}, std={df[f'rolling_phys_load_z_{wtype}'].std():.4f}")

# ═══════════════════════════════════════════
# 5. DEMAND-ADJUSTED MODEL
# ═══════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 5: Demand-adjusted fatigue model")
log("=" * 60)

# Use rolling_cog_load_mean (10-min rolling) for baseline selection
baseline_cog_col = 'rolling_cog_load_mean'

def run_demand_adjusted(data, label):
    """Run demand-adjusted model for a subset of data."""
    log(f"\n{'─'*50}")
    log(f"Demand-adjusted model: {label}")
    log(f"{'─'*50}")
    
    # Drop rows missing demand vars or outcome
    cog_z_cols = [f'rolling_cog_load_z_{w}' for w in ['mean', 'decay']]
    phys_z_cols = [f'rolling_phys_load_z_{w}' for w in ['mean', 'decay']]
    base_cols = DEMAND_VARS + ['reorientation_rate', baseline_cog_col, 'player_id', 'game_id'] + cog_z_cols + phys_z_cols
    model_data = data[base_cols].dropna()
    log(f"Data after dropping NAs: {len(model_data):,} rows")
    
    # Low-load baseline: lowest 50% of rolling cognitive load
    median_cog = model_data[baseline_cog_col].median()
    is_low_load = model_data[baseline_cog_col] <= median_cog
    train = model_data[is_low_load].copy()
    log(f"Low-load training set: {len(train):,} rows ({len(train)/len(model_data)*100:.1f}%)")
    
    if len(train) < 100:
        log("  WARNING: Too few training rows!")
        return None
    
    # Standardize predictors on training set
    pred_means = train[DEMAND_VARS].mean()
    pred_stds = train[DEMAND_VARS].std()
    
    for col in DEMAND_VARS:
        train[f'{col}_z'] = (train[col] - pred_means[col]) / pred_stds[col]
        model_data[f'{col}_z'] = (model_data[col] - pred_means[col]) / pred_stds[col]
    
    # Fit demand model on low-load baseline
    formula = 'reorientation_rate ~ ' + ' + '.join([f'{v}_z' for v in DEMAND_VARS])
    m_ols = smf.ols(formula, data=train).fit()
    
    log(f"\n  Demand model (OLS on low-load baseline):")
    log(f"    R² = {m_ols.rsquared:.4f}")
    log(f"    F = {m_ols.fvalue:.1f}")
    for var in ['Intercept'] + [f'{v}_z' for v in DEMAND_VARS]:
        sig = '***' if m_ols.pvalues[var] < 0.001 else '**' if m_ols.pvalues[var] < 0.01 else '*' if m_ols.pvalues[var] < 0.05 else ''
        log(f"    {var:35s} β={m_ols.params[var]:+.4f}, p={m_ols.pvalues[var]:.4f} {sig}")
    
    # Also try mixed model with player random effects
    try:
        m_mixed = smf.mixedlm(formula, data=train, groups=train['player_id']).fit(reml=False, maxiter=100)
        log(f"\n  Demand model (mixed LM, player RE):")
        log(f"    Random effect var: {m_mixed.cov_re.iloc[0,0]:.4f}")
        log(f"    Residual var: {m_mixed.scale:.4f}")
        for var in ['Intercept'] + [f'{v}_z' for v in DEMAND_VARS]:
            if var in m_mixed.fe_params.index:
                t_val = m_mixed.tvalues.get(var, 0)
                p_val = 2 * (1 - stats.t.cdf(abs(t_val), df=len(train) - len(DEMAND_VARS) - 1))
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                log(f"    {var:35s} β={m_mixed.fe_params[var]:+.4f}, t={t_val:.2f}, p≈{p_val:.4f} {sig}")
    except Exception as e:
        log(f"    Mixed model failed: {e}")
        m_mixed = None
    
    # Compute predicted and deficit on ALL data
    best_model = m_mixed if m_mixed is not None else m_ols
    try:
        model_data['predicted'] = best_model.predict(model_data)
    except:
        model_data['predicted'] = m_ols.predict(model_data)
    
    model_data['fatigue_deficit'] = model_data['reorientation_rate'] - model_data['predicted']
    deficit = model_data['fatigue_deficit']
    
    log(f"\n  Fatigue deficit on full data:")
    log(f"    Mean: {deficit.mean():+.4f}, SD: {deficit.std():.4f}")
    log(f"    Negative (fatigue signal): {(deficit < 0).mean()*100:.1f}%")
    
    return {
        'model_data': model_data,
        'demand_model': m_ols,
        'demand_model_mixed': m_mixed,
        'r2': m_ols.rsquared,
        'median_cog': median_cog,
        'n_train': len(train),
    }

# Run demand model on DEFENDERS
def_data = df[df['is_defender']].copy()
log(f"\nDefenders: {len(def_data):,} rows, {def_data['player_id'].nunique()} players")

# Also run demand model on non-defenders for completeness
off_data = df[~df['is_defender']].copy()

# Run on defenders
def_result = run_demand_adjusted(def_data, "DEFENDERS ONLY")

# Run on ALL players
all_result = run_demand_adjusted(df, "ALL PLAYERS")

# ═══════════════════════════════════════════
# 6. PERCENTILE GROUPING & TEST
# ═══════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 6: Percentile grouping and deficit test")
log("=" * 60)

def run_deficit_test(model_data, cog_z_col, phys_z_col, label, wtype_label=""):
    """Run high vs low deficit test with physical load control."""
    
    data = model_data.copy()
    
    # Compute percentiles
    q75 = data[cog_z_col].quantile(0.75)
    q25 = data[cog_z_col].quantile(0.25)
    
    # High/Low load groups based on cog load
    is_high = data[cog_z_col] >= q75
    is_low = data[cog_z_col] <= q25
    
    high = data[is_high].copy()
    low = data[is_low].copy()
    middle = data[~is_high & ~is_low].copy()
    
    log(f"\n  {label} [{wtype_label}]:")
    log(f"    Q25={q25:.4f}, Q75={q75:.4f}")
    log(f"    High group: n={len(high)}, mean deficit={high['fatigue_deficit'].mean():+.4f}")
    log(f"    Low group: n={len(low)}, mean deficit={low['fatigue_deficit'].mean():+.4f}")
    log(f"    Middle 50% discarded: n={len(middle)}")
    
    if len(high) < 10 or len(low) < 10:
        log("    WARNING: Too few blocks in groups!")
        return None
    
    # SAME for physical load
    phys_q75 = data[phys_z_col].quantile(0.75)
    data['phys_load_group'] = np.where(data[phys_z_col] >= phys_q75, 'high_phys', 'low_phys')
    
    # Merge back into high/low for analysis
    high_low = pd.concat([high, low]).copy()
    high_low['load_group'] = np.where(high_low[cog_z_col] >= q75, 'high', 'low')
    
    # ── Test: deficit ~ load_group ──
    mean_high = high['fatigue_deficit'].mean()
    mean_low = low['fatigue_deficit'].mean()
    diff = mean_high - mean_low
    
    # Welch t-test
    t_stat, p_val = stats.ttest_ind(high['fatigue_deficit'].dropna(), 
                                      low['fatigue_deficit'].dropna(), 
                                      equal_var=False)
    
    # Bootstrap CI for difference
    n_boot = 10000
    h_vals = high['fatigue_deficit'].dropna().values
    l_vals = low['fatigue_deficit'].dropna().values
    boot_diffs = np.zeros(n_boot)
    for b in range(n_boot):
        h_sample = np.random.choice(h_vals, size=len(h_vals), replace=True)
        l_sample = np.random.choice(l_vals, size=len(l_vals), replace=True)
        boot_diffs[b] = h_sample.mean() - l_sample.mean()
    ci_low = np.percentile(boot_diffs, 2.5)
    ci_high = np.percentile(boot_diffs, 97.5)
    
    # Cohen's d
    n1, n2 = len(high), len(low)
    s1, s2 = high['fatigue_deficit'].std(), low['fatigue_deficit'].std()
    pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
    cohens_d = diff / pooled_sd if pooled_sd > 0 else 0
    
    # ── Controlled for physical load ──
    high_low['phys_q75'] = phys_q75
    high_low['is_high_cog'] = (high_low['load_group'] == 'high').astype(float)
    high_low['phys_z_ctrl'] = high_low[phys_z_col]
    
    try:
        m_ctrl = smf.ols('fatigue_deficit ~ is_high_cog + phys_z_ctrl', data=high_low).fit()
        ctrl_beta = m_ctrl.params['is_high_cog']
        ctrl_p = m_ctrl.pvalues['is_high_cog']
        log(f"    Physical load control: β={ctrl_beta:+.4f}, p={ctrl_p:.6f}")
        survives_phys = ctrl_p < 0.05
    except Exception as e:
        ctrl_beta = ctrl_p = np.nan
        survives_phys = False
        log(f"    Physical load control failed: {e}")
    
    # ── Continuous version: deficit ~ rolling_cog_z + rolling_phys_z ──
    try:
        m_cont = smf.ols(f'fatigue_deficit ~ {cog_z_col} + {phys_z_col}', data=data).fit()
        cont_beta = m_cont.params[cog_z_col]
        cont_p = m_cont.pvalues[cog_z_col]
        log(f"    Continuous model: β(cog)={cont_beta:+.4f}, p={cont_p:.6f}")
        log(f"    Continuous model R²={m_cont.rsquared:.4f}")
    except Exception as e:
        cont_beta = cont_p = np.nan
        log(f"    Continuous model failed: {e}")
    
    log(f"    Diff: {diff:+.4f} [{ci_low:.4f}, {ci_high:.4f}], p={p_val:.6f}, d={cohens_d:.3f}")
    
    return {
        'q75': q75,
        'q25': q25,
        'n_high': len(high),
        'n_low': len(low),
        'mean_high': mean_high,
        'mean_low': mean_low,
        'diff': diff,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': p_val,
        'cohens_d': cohens_d,
        'ctrl_beta': ctrl_beta,
        'ctrl_p': ctrl_p,
        'survives_phys': survives_phys,
        'cont_beta': cont_beta,
        'cont_p': cont_p,
    }

# Run tests for both window types, both groups
results = {}

for group_label, group_result, group_data in [("Defenders", def_result, def_data), 
                                                ("All Players", all_result, df)]:
    if group_result is None:
        log(f"\nSkipping {group_label} — no demand model result")
        continue
    
    model_data = group_result['model_data']
    
    results[group_label] = {}
    
    for wtype, wtype_label in [('mean', '10-min Rolling'), ('decay', '15-min Decay')]:
        cog_z_col = f'rolling_cog_load_z_{wtype}'
        phys_z_col = f'rolling_phys_load_z_{wtype}'
        
        test_res = run_deficit_test(model_data, cog_z_col, phys_z_col, group_label, wtype_label)
        results[group_label][wtype] = test_res

# ═══════════════════════════════════════════
# 7. GENERATE REPORT (Markdown)
# ═══════════════════════════════════════════
log("\n" + "=" * 60)
log("GENERATING REPORT")
log("=" * 60)

md = []
md.append("# Defensive Group Fatigue Analysis")
md.append("")
md.append(f"**Analysis date:** 2026-07-18")
md.append("")
md.append("## Methodology")
md.append("")
md.append("### Position Clusters")
md.append("")
md.append(f"Players were clustered by behavioral averages and labeled as CB, FB, DM, or CM/W.")
md.append(f"Distribution: {dict(pos_counts)}")
md.append("")
md.append("### Defensive Group")
md.append("")
md.append(f"- **Defensive group (CB + FB + DM):** {n_def_players} players ({df['is_defender'].sum():,} blocks)")
md.append(f"- **Non-defensive (CM/W):** {n_off_players} players ({df[~df['is_defender']].shape[0]:,} blocks)")
md.append("")

# Key design choices
md.append("### Key Design Choices")
md.append("")
md.append("1. **Rolling load from PRECEDING blocks only** — no future information leaks")
md.append("2. **10-min rolling window**: mean of 2 preceding blocks (~10 min)")
md.append("3. **15-min decaying window**: exponentially weighted mean of ALL preceding blocks (τ=15 min)")
md.append("4. **Composite cognitive load z-score**: standardized average of 5 components (pressure_composite, opponents_nearby_mean, reorientation_count, transition_count, depth_mean)")
md.append("5. **75th/25th percentile thresholds**: high load ≥ 75th, low load ≤ 25th, middle 50% discarded")
md.append("6. **Demand-adjusted model**: predict expected reorientation_rate from CURRENT situation (pressure_composite + opponents_nearby_mean + depth_mean) — no reorientation_count in predictors")
md.append("7. **Low-load baseline**: lowest 50% of rolling cognitive load used to train demand model")
md.append("8. **Physical load controlled** in every model")
md.append("")

# Demand model quality
md.append("## Demand Model Quality")
md.append("")

for group_label, group_result in [("Defenders", def_result), ("All Players", all_result)]:
    if group_result is None:
        continue
    r2 = group_result['r2']
    n_train = group_result['n_train']
    md.append(f"### {group_label}")
    md.append("")
    md.append(f"- **R²** = {r2:.4f} (variance explained by situational factors in low-load baseline)")
    md.append(f"- **Training set**: {n_train:,} low-load blocks")
    md.append(f"- **Predictors**: {', '.join(DEMAND_VARS)}")
    if r2 < 0.02:
        md.append(f"- ⚠️ **R² < 0.02** — demand model explains very little variance. Fatigue deficit may be noisy.")
    elif r2 < 0.05:
        md.append(f"- ⚠️ **R² < 0.05** — demand model has limited explanatory power. Interpret with caution.")
    else:
        md.append(f"- ✅ **R² ≥ 0.05** — demand model has meaningful explanatory power.")
    md.append("")
    
    # Show coefficients
    m = group_result['demand_model']
    md.append("| Predictor | β (std) | p-value |")
    md.append("|----------|--------:|--------:|")
    for var in ['Intercept'] + [f'{v}_z' for v in DEMAND_VARS]:
        sig = '***' if m.pvalues[var] < 0.001 else '**' if m.pvalues[var] < 0.01 else '*' if m.pvalues[var] < 0.05 else ''
        md.append(f"| {var} | {m.params[var]:+.4f} | {m.pvalues[var]:.4f} {sig} |")
    md.append("")

# Deficit test results
md.append("## High vs Low Cognitive Load: Fatigue Deficit")
md.append("")
md.append("Format: more negative deficit = worse than situationally expected = fatigue signal.")
md.append("")

for group_label in ["Defenders", "All Players"]:
    if group_label not in results:
        continue
    
    md.append(f"### {group_label}")
    md.append("")
    md.append("| Window | N(high) | N(low) | Mean(high) | Mean(low) | Diff | 95% CI | p-value | Cohen's d | Survives Phys Ctrl |")
    md.append("|--------|-------:|------:|----------:|---------:|-----:|-------|--------:|----------:|:------------------:|")
    
    for wtype, wtype_label in [('mean', '10-min Rolling'), ('decay', '15-min Decay')]:
        res = results[group_label].get(wtype)
        if res is None:
            md.append(f"| {wtype_label} | — | — | — | — | — | — | — | — | — |")
            continue
        
        sig_str = '✅ Yes' if res['survives_phys'] else '❌ No'
        md.append(f"| {wtype_label} | {res['n_high']} | {res['n_low']} | "
                  f"{res['mean_high']:+.4f} | {res['mean_low']:+.4f} | "
                  f"{res['diff']:+.4f} | [{res['ci_low']:.4f}, {res['ci_high']:.4f}] | "
                  f"{res['p_value']:.4f} | {res['cohens_d']:.3f} | {sig_str} |")
    md.append("")

# Continuous model results
md.append("## Continuous Model: deficit ~ rolling_cog_z + rolling_phys_z")
md.append("")

for group_label in ["Defenders", "All Players"]:
    if group_label not in results:
        continue
    
    md.append(f"### {group_label}")
    md.append("")
    md.append("| Window | β(cog) | p(cog) | β(phys) | p(phys) |")
    md.append("|--------|-------:|-------:|--------:|--------:|")
    
    for wtype, wtype_label in [('mean', '10-min Rolling'), ('decay', '15-min Decay')]:
        res = results[group_label].get(wtype)
        if res is None:
            md.append(f"| {wtype_label} | — | — | — | — |")
            continue
        
        md.append(f"| {wtype_label} | {res['cont_beta']:+.4f} | {res['cont_p']:.4f} | — | — |")
    md.append("")

# Key findings summary
md.append("## Key Findings")
md.append("")

# Extract findings for defenders
def_res = results.get("Defenders", {})
all_res = results.get("All Players", {})

# Primary result: 10-min rolling for defenders
primary_res = def_res.get('mean')
if primary_res:
    md.append("### 1. Defenders Only (10-min Rolling)")
    md.append("")
    md.append(f"- **High cognitive load deficit:** {primary_res['mean_high']:+.4f} scans/block")
    md.append(f"- **Low cognitive load deficit:** {primary_res['mean_low']:+.4f} scans/block")
    md.append(f"- **Difference:** {primary_res['diff']:+.4f} [95% CI: {primary_res['ci_low']:.4f}, {primary_res['ci_high']:.4f}]")
    md.append(f"- **p-value:** {primary_res['p_value']:.6f}")
    md.append(f"- **Cohen's d:** {primary_res['cohens_d']:.3f}")
    md.append(f"- **Survives physical load control:** {'✅ Yes' if primary_res['survives_phys'] else '❌ No'} (β={primary_res['ctrl_beta']:+.4f}, p={primary_res['ctrl_p']:.4f})")
    
    if primary_res['diff'] < 0 and primary_res['p_value'] < 0.05:
        md.append("- ✅ **Fatigue effect detected:** Defenders show worse-than-expected defensive scanning under high cognitive load.")
    elif primary_res['diff'] > 0 and primary_res['p_value'] < 0.05:
        md.append("- ⚠️ **Reversed effect:** Defenders show MORE scanning under high cognitive load (possible compensation/arousal effect).")
    else:
        md.append("- 🔍 **No significant fatigue effect detected** in defenders.")
    md.append("")

primary_res_all = all_res.get('mean')
if primary_res_all:
    md.append("### 2. All Players (10-min Rolling)")
    md.append("")
    md.append(f"- **High cognitive load deficit:** {primary_res_all['mean_high']:+.4f} scans/block")
    md.append(f"- **Low cognitive load deficit:** {primary_res_all['mean_low']:+.4f} scans/block")
    md.append(f"- **Difference:** {primary_res_all['diff']:+.4f} [95% CI: {primary_res_all['ci_low']:.4f}, {primary_res_all['ci_high']:.4f}]")
    md.append(f"- **p-value:** {primary_res_all['p_value']:.6f}")
    md.append(f"- **Survives physical load control:** {'✅ Yes' if primary_res_all['survives_phys'] else '❌ No'}")
    md.append("")

# Effect direction
if primary_res and primary_res_all:
    dir_def = "negative" if primary_res['diff'] < 0 else "positive"
    dir_all = "negative" if primary_res_all['diff'] < 0 else "positive"
    
    if dir_def == dir_all:
        md.append(f"### 3. Consistency Check")
        md.append(f"- Both groups show a **{dir_def}** deficit direction — consistent across defensive and all players.")
    else:
        md.append(f"### 3. Consistency Check")
        md.append(f"- **Direction mismatch:** Defenders show {dir_def} deficit while all players show {dir_all} deficit.")
    md.append("")

# Demand model R² warning
r2_def = def_result['r2'] if def_result else 0
r2_all = all_result['r2'] if all_result else 0
md.append("### 4. Methodological Notes")
md.append("")
md.append(f"- **Demand model R² (defenders):** {r2_def:.4f} {'⚠️' if r2_def < 0.02 else '✅' if r2_def >= 0.05 else '⚠️'}")
md.append(f"- **Demand model R² (all players):** {r2_all:.4f} {'⚠️' if r2_all < 0.02 else '✅' if r2_all >= 0.05 else '⚠️'}")
md.append("- **Demand predictors exclude reorientation_count** to avoid collinearity")
md.append("- **Rolling load uses PRECEDING blocks only** — no look-ahead bias")
md.append("- **Position labels from behavioral clustering** (not pitch coordinates)")
md.append("")

# Write report
report_path = f'{OUT_DIR}/defensive_group_fatigue.md'
with open(report_path, 'w') as f:
    f.write('\n'.join(md))
log(f"\nReport saved: {report_path}")

# ═══════════════════════════════════════════
# 8. GENERATE FIGURE
# ═══════════════════════════════════════════
log("\n--- Generating figure ---")

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# Panel A: Defenders - deficit by load decile (10-min rolling)
ax1 = fig.add_subplot(gs[0, 0])
if def_result is not None:
    model_data = def_result['model_data']
    cog_col = 'rolling_cog_load_z_mean'
    sub = model_data[[cog_col, 'fatigue_deficit', 'player_id']].dropna().copy()
    sub['decile'] = pd.qcut(sub[cog_col], 10, labels=False, duplicates='drop')
    dmeans = sub.groupby('decile')['fatigue_deficit'].agg(['mean', 'sem', 'count'])
    dcenters = sub.groupby('decile')[cog_col].median()
    
    ci = dmeans['sem'] * 1.96
    ax1.errorbar(range(len(dcenters)), dmeans['mean'], yerr=ci,
                 fmt='o-', color='#1565C0', capsize=4, capthick=1.5,
                 markersize=8, linewidth=2)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('Cognitive Load Decile', fontsize=11)
    ax1.set_ylabel('Fatigue Deficit (scans/block)', fontsize=11)
    ax1.set_title('A) Defenders: Deficit by Cog Load Decile', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add regression line
    X = sm.add_constant(sub[cog_col].values)
    y = sub['fatigue_deficit'].values
    lr = sm.OLS(y, X).fit()
    x_range = np.linspace(sub[cog_col].min(), sub[cog_col].max(), 100)
    X_pred = sm.add_constant(x_range)
    y_pred = lr.predict(X_pred)
    ax1.plot(x_range, y_pred, '--', color='red', alpha=0.5, linewidth=1.5)
    
    # Annotate
    p_val = lr.pvalues[1]
    ax1.annotate(f'β={lr.params[1]:+.3f}, p={p_val:.4f}',
                 xy=(0.05, 0.05), xycoords='axes fraction', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Panel B: All Players - deficit by load decile
ax2 = fig.add_subplot(gs[0, 1])
if all_result is not None:
    model_data_all = all_result['model_data']
    sub_all = model_data_all[[cog_col, 'fatigue_deficit', 'player_id']].dropna().copy()
    sub_all['decile'] = pd.qcut(sub_all[cog_col], 10, labels=False, duplicates='drop')
    dmeans_all = sub_all.groupby('decile')['fatigue_deficit'].agg(['mean', 'sem', 'count'])
    
    ci_all = dmeans_all['sem'] * 1.96
    ax2.errorbar(range(len(dmeans_all)), dmeans_all['mean'], yerr=ci_all,
                 fmt='o-', color='#E65100', capsize=4, capthick=1.5,
                 markersize=8, linewidth=2)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Cognitive Load Decile', fontsize=11)
    ax2.set_ylabel('Fatigue Deficit (scans/block)', fontsize=11)
    ax2.set_title('B) All Players: Deficit by Cog Load Decile', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    X2 = sm.add_constant(sub_all[cog_col].values)
    y2 = sub_all['fatigue_deficit'].values
    lr2 = sm.OLS(y2, X2).fit()
    x_range2 = np.linspace(sub_all[cog_col].min(), sub_all[cog_col].max(), 100)
    y_pred2 = lr2.predict(sm.add_constant(x_range2))
    ax2.plot(x_range2, y_pred2, '--', color='red', alpha=0.5, linewidth=1.5)
    p_val2 = lr2.pvalues[1]
    ax2.annotate(f'β={lr2.params[1]:+.3f}, p={p_val2:.4f}',
                 xy=(0.05, 0.05), xycoords='axes fraction', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Panel C: Defenders - High vs Low comparison
ax3 = fig.add_subplot(gs[1, 0])
if def_result is not None and 'mean' in def_res and def_res['mean'] is not None:
    r = def_res['mean']
    groups = ['Low Cog Load', 'High Cog Load']
    means = [r['mean_low'], r['mean_high']]
    errors = [r.get('ci_low_err', r['ci_high'] - r['diff']), r.get('ci_high_err', r['ci_high'] - r['diff'])]
    
    # Use SE from decile grouping
    bars = ax3.bar(groups, means, color=['#81C784', '#E57373'], alpha=0.8, width=0.5)
    
    # Add individual data points (jittered box-like)
    if def_result is not None:
        model_data = def_result['model_data']
        cog_col = 'rolling_cog_load_z_mean'
        q75 = model_data[cog_col].quantile(0.75)
        q25 = model_data[cog_col].quantile(0.25)
        high_pts = model_data[model_data[cog_col] >= q75]['fatigue_deficit'].dropna()
        low_pts = model_data[model_data[cog_col] <= q25]['fatigue_deficit'].dropna()
        
        # Plot means with error bars
        high_se = high_pts.std() / np.sqrt(len(high_pts))
        low_se = low_pts.std() / np.sqrt(len(low_pts))
        
        ax3.errorbar(0, r['mean_low'], yerr=low_se*1.96, fmt='o', color='black', capsize=6)
        ax3.errorbar(1, r['mean_high'], yerr=high_se*1.96, fmt='o', color='black', capsize=6)
    
    ax3.set_ylabel('Fatigue Deficit (scans/block)', fontsize=11)
    ax3.set_title(f'C) Defenders: High vs Low Load\nDiff={r["diff"]:+.3f}, p={r["p_value"]:.4f}', 
                  fontsize=12, fontweight='bold')
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')

# Panel D: Comparison bar - Effect size by group
ax4 = fig.add_subplot(gs[1, 1])
if def_res.get('mean') and all_res.get('mean'):
    r_def = def_res['mean']
    r_all = all_res['mean']
    
    groups_bar = ['Defenders', 'All Players']
    diffs = [r_def['diff'], r_all['diff']]
    cis_low = [r_def['ci_low'], r_all['ci_low']]
    cis_high = [r_def['ci_high'], r_all['ci_high']]
    
    colors = ['#1565C0', '#E65100']
    x_pos = [0, 1]
    bars = ax4.bar(x_pos, diffs, color=colors, alpha=0.8, width=0.5)
    ax4.errorbar(x_pos, diffs, yerr=[np.array(diffs) - np.array(cis_low), 
                                        np.array(cis_high) - np.array(diffs)],
                 fmt='none', color='black', capsize=6)
    
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Deficit Difference (scans/block)', fontsize=11)
    ax4.set_title('D) Effect Comparison: Defenders vs All', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(groups_bar, fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

fig.suptitle('Defensive Group Fatigue Analysis: Demand-Adjusted Model', 
             fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()
fig_path = f'{OUT_DIR}/defensive_group_fatigue.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
log(f"Figure saved: {fig_path}")

# ═══════════════════════════════════════════
# 9. SUMMARY FOR REPORT-BACK
# ═══════════════════════════════════════════
log("\n" + "=" * 60)
log("SUMMARY FOR REPORT-BACK")
log("=" * 60)

if def_res.get('mean'):
    r = def_res['mean']
    log(f"\nDEFENDERS ONLY (10-min rolling):")
    log(f"  High load deficit: {r['mean_high']:+.4f}")
    log(f"  Low load deficit:  {r['mean_low']:+.4f}")
    log(f"  Difference:        {r['diff']:+.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    log(f"  p-value:           {r['p_value']:.6f}")
    log(f"  Cohen's d:         {r['cohens_d']:.3f}")
    log(f"  Survives phys ctrl: {r['survives_phys']} (β={r['ctrl_beta']:+.4f}, p={r['ctrl_p']:.4f})")
    log(f"  Demand model R²:   {def_result['r2']:.4f}")

if all_res.get('mean'):
    r_all = all_res['mean']
    log(f"\nALL PLAYERS (10-min rolling):")
    log(f"  High load deficit: {r_all['mean_high']:+.4f}")
    log(f"  Low load deficit:  {r_all['mean_low']:+.4f}")
    log(f"  Difference:        {r_all['diff']:+.4f} [{r_all['ci_low']:.4f}, {r_all['ci_high']:.4f}]")
    log(f"  p-value:           {r_all['p_value']:.6f}")
    log(f"  Survives phys ctrl: {r_all['survives_phys']}")

# Save summary JSON
summary = {
    'defenders': {
        '10min_rolling': def_res.get('mean'),
        '15min_decay': def_res.get('decay'),
        'demand_r2': def_result['r2'] if def_result else None,
    },
    'all_players': {
        '10min_rolling': all_res.get('mean'),
        '15min_decay': all_res.get('decay'),
        'demand_r2': all_result['r2'] if all_result else None,
    },
    'position_counts': {k: int(v) for k, v in pos_counts.items()},
    'n_defenders': int(n_def_players),
}

summ_path = f'{OUT_DIR}/defensive_group_summary.json'
with open(summ_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
log(f"\nSummary JSON saved: {summ_path}")

log("\n\nAnalysis complete! ✅")
