#!/usr/bin/env python3
"""
Clean Percentile Fatigue Analysis — Exclude First Blocks
=========================================================
Problem: First 1-2 blocks per player-game have zero accumulated rolling load
(preceding blocks), which artificially pulls the "low load" group toward a 
baseline of fresh-start performance.

Fix: Exclude blocks where rolling cognitive load is at floor (first 2 blocks
per player per game). Rerun defensive-group comparison on clean subsets.

Steps:
1. Load positions from lookup
2. Compute rolling fatigue (preceding blocks only)
3. Remove contaminated blocks (first 2 per player-game)
4. Defensive group only (CB + FB + DM)
5. Demand model on clean low-load blocks
6. Continuous model: fatigue_deficit ~ rolling_cog_load_z + rolling_phys_load_z
7. Percentile split on clean data
8. Compare: first blocks included vs excluded
"""

import sys, warnings, json
import pandas as pd
import numpy as np
from scipy import stats
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

BLOCK_MINUTES = 5.0
TAU = 15.0  # 15-min decay constant

# Cognitive load indicators (for z-score composite)
COG_INDICATORS = [
    'pressure_composite',
    'opponents_nearby_mean',
    'reorientation_count',
    'transition_count',
    'depth_mean',
]

# Demand predictors: situation only, NO reorientation_count
DEMAND_VARS = [
    'pressure_composite',
    'opponents_nearby_mean',
    'depth_mean',
]

PHYS_VAR = 'physical_load'
OUTCOME = 'reorientation_rate'
# How many blocks to remove from start per player per game
N_REMOVE_BLOCKS = 2

log("=" * 70)
log("CLEAN PERCENTILE FATIGUE ANALYSIS")
log("Exclude first blocks from percentile split")
log("=" * 70)

# ═══════════════════════════════════════════
# 1. LOAD DATA & POSITIONS
# ═══════════════════════════════════════════
log("\n--- Step 1: Load data ---")
df = pd.read_parquet(DATA_PATH)
log(f"Loaded {len(df):,} rows, {df['player_id'].nunique()} players, {df['game_id'].nunique()} games")

lookup = pd.read_csv(LOOKUP_PATH)
log(f"Position lookup: {len(lookup)} players mapped")

df = df.merge(lookup, on='player_id', how='left')
log(f"After merge: {df['position'].isna().sum()} players with no position mapping")

# Position counts
pos_counts = df.groupby('position')['player_id'].nunique()
for pos, cnt in pos_counts.items():
    log(f"  {pos}: {cnt} players")

# ═══════════════════════════════════════════
# 2. DEFINE DEFENSIVE GROUP
# ═══════════════════════════════════════════
log("\n--- Step 2: Define defensive group ---")
df['is_defender'] = df['position'].isin(['CB', 'FB', 'DM'])
n_def = df[df['is_defender']]['player_id'].nunique()
n_off = df[~df['is_defender']]['player_id'].nunique()
log(f"Defensive (CB+FB+DM): {n_def} players ({df['is_defender'].sum():,} blocks)")
log(f"Non-defensive (CM/W): {n_off} players ({df[~df['is_defender']].shape[0]:,} blocks)")

# Sort for within-game sequencing
df = df.sort_values(['game_id', 'player_id', 'phase', 'block_num']).reset_index(drop=True)
df['pg_key'] = df['player_id'].astype(str) + '_' + df['game_id'].astype(str)

# ═══════════════════════════════════════════
# 3. COMPUTE ROLLING COGNITIVE & PHYSICAL LOAD
# ═══════════════════════════════════════════
log("\n--- Step 3: Compute rolling load from PRECEDING blocks ---")

# Standardize indicators first (pooled z-score)
for col in COG_INDICATORS:
    mu, sigma = df[col].mean(), df[col].std()
    df[f'{col}_z'] = (df[col] - mu) / sigma

# Instantaneous cognitive load composite
df['cog_load'] = df[[f'{c}_z' for c in COG_INDICATORS]].mean(axis=1)

# Also standardize physical load
phys_mean, phys_std = df[PHYS_VAR].mean(), df[PHYS_VAR].std()
df['phys_load_z'] = (df[PHYS_VAR] - phys_mean) / phys_std

def ewma(values, tau_blk=3.0):
    """Exponentially weighted moving average. tau in block units (1 block = 5 min)."""
    if len(values) == 0:
        return np.nan
    alpha = 1 - np.exp(-1 / tau_blk) if tau_blk > 0 else 1.0
    n = len(values)
    weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
    weights = weights / weights.sum()
    return np.average(values, weights=weights)

# Initialize columns
df['rolling_cog_load_mean'] = np.nan      # 10-min rolling = mean of 2 preceding blocks
df['rolling_cog_load_decay'] = np.nan     # 15-min exp decay
df['rolling_phys_load_mean'] = np.nan
df['rolling_phys_load_decay'] = np.nan
df['n_preceding_blocks'] = 0

# Track for contamination analysis: which blocks are "first N"
df['is_first_block'] = False
df['is_contaminated'] = False

groups = list(df.groupby('pg_key', sort=False))
n_groups = len(groups)

for gi, (key, grp) in enumerate(groups):
    if gi % 500 == 0:
        log(f"  Group {gi}/{n_groups}...")
    
    # CRITICAL: capture original indices BEFORE reset_index
    grp_sorted = grp.sort_values(['phase', 'block_num'])
    orig_indices = grp_sorted.index.values  # real df indices
    n = len(grp_sorted)
    
    cog_vals = grp_sorted['cog_load'].values
    phys_vals = grp_sorted['phys_load_z'].values
    phase_vals = grp_sorted['phase'].values
    
    for i in range(n):
        idx = orig_indices[i]  # the real row index in df
        preceding = np.arange(i)  # indices 0..i-1 within group
        n_prec = len(preceding)
        
        df.loc[idx, 'n_preceding_blocks'] = n_prec
        df.loc[idx, 'is_first_block'] = (i < 1)  # exact first block
        df.loc[idx, 'is_contaminated'] = (i < N_REMOVE_BLOCKS)  # first N blocks
        
        if n_prec == 0:
            # First block: no preceding data -> NaN for rolling
            df.loc[idx, 'rolling_cog_load_mean'] = np.nan
            df.loc[idx, 'rolling_cog_load_decay'] = np.nan
            df.loc[idx, 'rolling_phys_load_mean'] = np.nan
            df.loc[idx, 'rolling_phys_load_decay'] = np.nan
            continue
        
        # --- 10-min rolling: preceding 2 blocks ---
        window_size = min(2, n_prec)
        w_preceding = np.arange(i - window_size, i)
        
        df.loc[idx, 'rolling_cog_load_mean'] = np.mean(cog_vals[w_preceding])
        df.loc[idx, 'rolling_phys_load_mean'] = np.mean(phys_vals[w_preceding])
        
        # --- 15-min exp decay: ALL preceding ---
        tau_blk = 3.0  # tau=15 min / 5 min per block
        df.loc[idx, 'rolling_cog_load_decay'] = ewma(cog_vals[preceding], tau_blk=tau_blk)
        df.loc[idx, 'rolling_phys_load_decay'] = ewma(phys_vals[preceding], tau_blk=tau_blk)

log("Rolling load computed.")

# Z-score the rolling measures
for wtype in ['mean', 'decay']:
    for var_base in ['cog', 'phys']:
        col = f'rolling_{var_base}_load_{wtype}'
        sub = df[col].dropna()
        if len(sub) > 0:
            mu, sigma = sub.mean(), sub.std()
            df[f'{col}_z'] = (df[col] - mu) / sigma if sigma > 0 else 0.0

# Rolling cog load using z-score composite (same approach as original percentile model)
# We already have the cog_load composite z-scored, so rolling of that = rolling cog load
for wtype in ['mean', 'decay']:
    col = f'rolling_cog_load_{wtype}'
    df[f'rolling_cog_load_z_{wtype}'] = df[f'{col}_z']  # already z-scored above

# For reporting - use the mean window as primary
PRIMARY_COG = 'rolling_cog_load_mean_z'
PRIMARY_PHYS = 'rolling_phys_load_mean_z'
PRIMARY_COG_RAW = 'rolling_cog_load_mean'
PRIMARY_PHYS_RAW = 'rolling_phys_load_mean'

log(f"\nRolling coverage:")
for col in ['rolling_cog_load_mean', 'rolling_cog_load_decay', 'rolling_cog_load_mean_z']:
    nn = df[col].notna().sum()
    log(f"  {col}: {nn:,}/{len(df):,} ({nn/len(df)*100:.1f}%)")

# ═══════════════════════════════════════════
# 4. BUILD CLEAN AND FULL DATASETS
# ═══════════════════════════════════════════
log("\n--- Step 4: Build clean dataset ---")

# FULL dataset: drop only NaN rolling (just the very first block)
df_full = df.dropna(subset=[PRIMARY_COG_RAW]).copy()
log(f"Full dataset (after dropping 1st-block NaN): {len(df_full):,} blocks")

# CLEAN dataset: also remove first N blocks per player-game
df_clean = df[~df['is_contaminated']].dropna(subset=[PRIMARY_COG_RAW]).copy()
log(f"Clean dataset (first {N_REMOVE_BLOCKS} blocks removed): {len(df_clean):,} blocks")
log(f"  Removed {len(df_full) - len(df_clean):,} contaminated blocks")

# Filter to defensive only
def_full = df_full[df_full['is_defender']].copy()
def_clean = df_clean[df_clean['is_defender']].copy()
log(f"Defenders full: {len(def_full):,} blocks from {def_full['player_id'].nunique()} players")
log(f"Defenders clean: {len(def_clean):,} blocks from {def_clean['player_id'].nunique()} players")

# ═══════════════════════════════════════════
# 5. DEMAND MODEL ON CLEAN LOW-LOAD BASELINE
# ═══════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 5: Demand-adjusted model (clean low-load baseline)")
log("=" * 60)

def run_demand_model(data, label, use_clean=True):
    """Fit demand model on low-load blocks, compute fatigue deficit."""
    log(f"\nDemand model: {label} (clean={use_clean})")
    
    # Include rolling z-score columns so downstream models can use them
    rolling_z_cols = ['rolling_cog_load_mean_z', 'rolling_cog_load_decay_z',
                      'rolling_phys_load_mean_z', 'rolling_phys_load_decay_z']
    available_z = [c for c in rolling_z_cols if c in data.columns]
    base_cols = DEMAND_VARS + [OUTCOME, PRIMARY_COG_RAW, 'player_id', 'game_id'] + available_z
    model_data = data[base_cols].dropna().copy()
    log(f"  N after dropping NAs: {len(model_data):,}")
    
    # Low-load baseline: bottom 50% of rolling cognitive load
    median_cog = model_data[PRIMARY_COG_RAW].median()
    is_low = model_data[PRIMARY_COG_RAW] <= median_cog
    train = model_data[is_low].copy()
    log(f"  Low-load training set: {len(train):,} ({len(train)/len(model_data)*100:.1f}%)")
    
    if len(train) < 100:
        log(f"  WARNING: Too few training rows!")
        return None
    
    # Standardize predictors
    pred_means = train[DEMAND_VARS].mean()
    pred_stds = train[DEMAND_VARS].std()
    for col in DEMAND_VARS:
        train[f'{col}_z'] = (train[col] - pred_means[col]) / pred_stds[col]
        model_data[f'{col}_z'] = (model_data[col] - pred_means[col]) / pred_stds[col]
    
    # Fit on low-load baseline
    formula = f'{OUTCOME} ~ ' + ' + '.join([f'{v}_z' for v in DEMAND_VARS])
    m_ols = smf.ols(formula, data=train).fit()
    
    log(f"  Demand model R² = {m_ols.rsquared:.4f}")
    for var in ['Intercept'] + [f'{v}_z' for v in DEMAND_VARS]:
        sig = '***' if m_ols.pvalues[var] < 0.001 else '**' if m_ols.pvalues[var] < 0.01 else '*' if m_ols.pvalues[var] < 0.05 else ''
        log(f"    {var:30s}: β={m_ols.params[var]:+.4f}, p={m_ols.pvalues[var]:.4f} {sig}")
    
    # Also try mixed model
    try:
        m_mixed = smf.mixedlm(formula, data=train, groups=train['player_id']).fit(reml=False, maxiter=100)
        log(f"  Mixed model: player RE var={m_mixed.cov_re.iloc[0,0]:.4f}")
        m_best = m_mixed
    except Exception as e:
        log(f"  Mixed model failed: {e}")
        m_best = m_ols
    
    # Predict on ALL blocks (not just clean, but the data we passed)
    try:
        model_data['predicted'] = m_best.predict(model_data)
    except:
        model_data['predicted'] = m_ols.predict(model_data)
    
    model_data['fatigue_deficit'] = model_data[OUTCOME] - model_data['predicted']
    deficit = model_data['fatigue_deficit']
    
    log(f"  Fatigue deficit: mean={deficit.mean():+.4f}, SD={deficit.std():.4f}")
    log(f"  Negative (fatigue signal): {(deficit < 0).mean()*100:.1f}%")
    
    return {
        'model_data': model_data,
        'ols': m_ols,
        'mixed': m_best,
        'r2': m_ols.rsquared,
        'n_train': len(train),
        'median_cog': median_cog,
    }

# Run demand model on clean defenders
def_result_clean = run_demand_model(def_clean, "Defenders CLEAN", use_clean=True)
# Also run on full defenders for comparison
def_result_full = run_demand_model(def_full, "Defenders FULL", use_clean=False)

# ═══════════════════════════════════════════
# 6. CONTINUOUS MODEL (PRIMARY)
# ═══════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 6: Continuous model — fatigue_deficit ~ rolling_cog_z + rolling_phys_z")
log("=" * 60)

def run_continuous_model(model_data, cog_col, phys_col, label):
    """Run continuous regression: fatigue_deficit ~ rolling_cog_load_z + rolling_phys_load_z."""
    sub = model_data[[cog_col, phys_col, 'fatigue_deficit', 'player_id', 'game_id']].dropna().copy()
    log(f"\n  {label}:")
    log(f"    N = {len(sub):,}")
    
    if len(sub) < 100:
        log("    WARNING: insufficient data")
        return None
    
    # OLS
    m_ols = smf.ols(f'fatigue_deficit ~ {cog_col} + {phys_col}', data=sub).fit()
    
    β_cog = m_ols.params[cog_col]
    se_cog = m_ols.bse[cog_col]
    ci_cog_low = β_cog - 1.96 * se_cog
    ci_cog_high = β_cog + 1.96 * se_cog
    t_cog = m_ols.tvalues[cog_col]
    p_cog = m_ols.pvalues[cog_col]
    
    β_phys = m_ols.params[phys_col]
    se_phys = m_ols.bse[phys_col]
    ci_phys_low = β_phys - 1.96 * se_phys
    ci_phys_high = β_phys + 1.96 * se_phys
    t_phys = m_ols.tvalues[phys_col]
    p_phys = m_ols.pvalues[phys_col]
    
    r2 = m_ols.rsquared
    r2_adj = m_ols.rsquared_adj
    
    sig_cog = '***' if p_cog < 0.001 else '**' if p_cog < 0.01 else '*' if p_cog < 0.05 else 'ns'
    sig_phys = '***' if p_phys < 0.001 else '**' if p_phys < 0.01 else '*' if p_phys < 0.05 else 'ns'
    
    log(f"    Cognitive load: β={β_cog:+.6f} [95% CI: {ci_cog_low:.6f}, {ci_cog_high:.6f}], t={t_cog:.2f}, p={p_cog:.6f} {sig_cog}")
    log(f"    Physical load:  β={β_phys:+.6f} [95% CI: {ci_phys_low:.6f}, {ci_phys_high:.6f}], t={t_phys:.2f}, p={p_phys:.6f} {sig_phys}")
    log(f"    R² = {r2:.4f}, R²_adj = {r2_adj:.4f}")
    
    # Direction check
    direction = "negative (fatigue)" if β_cog < 0 and p_cog < 0.05 else \
                "positive (compensation)" if β_cog > 0 and p_cog < 0.05 else \
                "not significant"
    log(f"    Direction: {direction}")
    
    return {
        'β_cog': β_cog, 'se_cog': se_cog, 'ci_cog_low': ci_cog_low, 'ci_cog_high': ci_cog_high,
        't_cog': t_cog, 'p_cog': p_cog, 'sig_cog': sig_cog,
        'β_phys': β_phys, 'se_phys': se_phys, 'ci_phys_low': ci_phys_low, 'ci_phys_high': ci_phys_high,
        't_phys': t_phys, 'p_phys': p_phys, 'sig_phys': sig_phys,
        'r2': r2, 'r2_adj': r2_adj, 'n': len(sub),
        'direction': direction,
    }

# Also run mixed model for robustness check
def run_continuous_mixed(model_data, cog_col, phys_col, label):
    """Mixed model with player random effects."""
    sub = model_data[[cog_col, phys_col, 'fatigue_deficit', 'player_id']].dropna().copy()
    if len(sub) < 100:
        return None
    try:
        m_mixed = smf.mixedlm(f'fatigue_deficit ~ {cog_col} + {phys_col}', 
                              data=sub, groups=sub['player_id'],
                              re_formula='1').fit(reml=False, maxiter=100)
        β_cog = m_mixed.fe_params[cog_col]
        t_cog = m_mixed.tvalues.get(cog_col, 0)
        p_cog = 2 * (1 - stats.t.cdf(abs(t_cog), df=len(sub) - 3))
        β_phys = m_mixed.fe_params[phys_col]
        t_phys = m_mixed.tvalues.get(phys_col, 0)
        p_phys = 2 * (1 - stats.t.cdf(abs(t_phys), df=len(sub) - 3))
        
        log(f"  Mixed model {label}: β(cog)={β_cog:+.6f}, p≈{p_cog:.6f}, β(phys)={β_phys:+.6f}, p≈{p_phys:.6f}")
        return {'β_cog': β_cog, 'p_cog': p_cog, 'β_phys': β_phys, 'p_phys': p_phys}
    except Exception as e:
        log(f"  Mixed model failed: {e}")
        return None

cont_results = {}

# Clean defenders
if def_result_clean:
    md = def_result_clean['model_data']
    cont_clean = run_continuous_model(md, 'rolling_cog_load_mean_z', 'rolling_phys_load_mean_z', "Defenders CLEAN (10-min rolling)")
    cont_decay_clean = run_continuous_model(md, 'rolling_cog_load_decay_z', 'rolling_phys_load_decay_z', "Defenders CLEAN (15-min decay)")
    cont_mixed_clean = run_continuous_mixed(md, 'rolling_cog_load_mean_z', 'rolling_phys_load_mean_z', "Defenders CLEAN")
    cont_results['clean'] = {'10min': cont_clean, '15min_decay': cont_decay_clean, 'mixed': cont_mixed_clean}

# Full defenders (with contaminated blocks)
if def_result_full:
    md_full = def_result_full['model_data']
    cont_full = run_continuous_model(md_full, 'rolling_cog_load_mean_z', 'rolling_phys_load_mean_z', "Defenders FULL (10-min rolling)")
    cont_decay_full = run_continuous_model(md_full, 'rolling_cog_load_decay_z', 'rolling_phys_load_decay_z', "Defenders FULL (15-min decay)")
    cont_mixed_full = run_continuous_mixed(md_full, 'rolling_cog_load_mean_z', 'rolling_phys_load_mean_z', "Defenders FULL")
    cont_results['full'] = {'10min': cont_full, '15min_decay': cont_decay_full, 'mixed': cont_mixed_full}

# ═══════════════════════════════════════════
# 7. PERCENTILE SPLIT (CLEAN, SECONDARY)
# ═══════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 7: Percentile split on clean data")
log("=" * 60)

def run_percentile_test(model_data, cog_col, phys_col, label, pct_thresholds=(0.25, 0.75)):
    """High vs Low cognitive load comparison with physical load control."""
    
    sub = model_data[[cog_col, phys_col, 'fatigue_deficit', 'player_id', 'game_id']].dropna().copy()
    log(f"\n  {label}:")
    log(f"    N total: {len(sub):,}")
    
    if len(sub) < 100:
        log(f"    WARNING: insufficient data")
        return None
    
    # Percentile thresholds from the clean subset
    p25 = sub[cog_col].quantile(pct_thresholds[0])
    p75 = sub[cog_col].quantile(pct_thresholds[1])
    
    # Physical load percentiles
    phys_p25 = sub[phys_col].quantile(pct_thresholds[0])
    phys_p75 = sub[phys_col].quantile(pct_thresholds[1])
    
    # Assign groups
    is_high = sub[cog_col] >= p75
    is_low = sub[cog_col] <= p25
    is_mid = ~is_high & ~is_low
    
    high = sub[is_high].copy()
    low = sub[is_low].copy()
    middle = sub[is_mid].copy()
    
    # Physical load groups
    sub['phys_high'] = (sub[phys_col] >= phys_p75).astype(float)
    
    log(f"    Percentiles: p25={p25:.4f}, p75={p75:.4f}")
    log(f"    High: n={len(high)}, mean deficit={high['fatigue_deficit'].mean():+.4f}")
    log(f"    Low:  n={len(low)}, mean deficit={low['fatigue_deficit'].mean():+.4f}")
    log(f"    Middle (discarded): n={len(middle)}")
    
    if len(high) < 10 or len(low) < 10:
        log("    WARNING: Too few blocks in groups")
        return None
    
    # Means and CIs
    mean_high = high['fatigue_deficit'].mean()
    mean_low = low['fatigue_deficit'].mean()
    se_high = high['fatigue_deficit'].sem()
    se_low = low['fatigue_deficit'].sem()
    
    ci_high_low = mean_high - 1.96 * se_high
    ci_high_high = mean_high + 1.96 * se_high
    ci_low_low = mean_low - 1.96 * se_low
    ci_low_high = mean_low + 1.96 * se_low
    
    # Univariate difference
    diff = mean_high - mean_low
    se_diff = np.sqrt(se_high**2 + se_low**2)
    ci_diff_low = diff - 1.96 * se_diff
    ci_diff_high = diff + 1.96 * se_diff
    
    t_stat, p_val = stats.ttest_ind(high['fatigue_deficit'].dropna(), 
                                     low['fatigue_deficit'].dropna(), 
                                     equal_var=False)
    
    # Bootstrap CI
    n_boot = 10000
    h_vals = high['fatigue_deficit'].dropna().values
    l_vals = low['fatigue_deficit'].dropna().values
    boot_diffs = np.zeros(n_boot)
    for b in range(n_boot):
        h_sample = np.random.choice(h_vals, size=len(h_vals), replace=True)
        l_sample = np.random.choice(l_vals, size=len(l_vals), replace=True)
        boot_diffs[b] = h_sample.mean() - l_sample.mean()
    boot_ci_low = np.percentile(boot_diffs, 2.5)
    boot_ci_high = np.percentile(boot_diffs, 97.5)
    
    # Cohen's d
    n1, n2 = len(high), len(low)
    s1, s2 = high['fatigue_deficit'].std(ddof=1), low['fatigue_deficit'].std(ddof=1)
    pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
    cohens_d = diff / pooled_sd if pooled_sd > 0 else 0
    
    sig_txt = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    
    # Controlled for physical load
    high_low_concat = pd.concat([high, low]).copy()
    high_low_concat['is_high_cog'] = (high_low_concat.index.isin(high.index)).astype(float)
    
    try:
        m_ctrl = smf.ols(f'fatigue_deficit ~ is_high_cog + {phys_col}', data=high_low_concat).fit()
        ctrl_beta = m_ctrl.params['is_high_cog']
        ctrl_se = m_ctrl.bse['is_high_cog']
        ctrl_ci_low = ctrl_beta - 1.96 * ctrl_se
        ctrl_ci_high = ctrl_beta + 1.96 * ctrl_se
        ctrl_p = m_ctrl.pvalues['is_high_cog']
        ctrl_sig = "***" if ctrl_p < 0.001 else "**" if ctrl_p < 0.01 else "*" if ctrl_p < 0.05 else "ns"
        log(f"    Controlled for phys load: β={ctrl_beta:+.6f} [95% CI: {ctrl_ci_low:.6f}, {ctrl_ci_high:.6f}], p={ctrl_p:.6f} {ctrl_sig}")
    except Exception as e:
        ctrl_beta = ctrl_p = ctrl_ci_low = ctrl_ci_high = np.nan
        ctrl_sig = ""
        log(f"    Physical load control failed: {e}")
    
    # Same with both phys load group
    high_low_concat['phys_high_bool'] = high_low_concat[phys_col] >= phys_p75
    try:
        m_ctrl2 = smf.ols(f'fatigue_deficit ~ is_high_cog + phys_high_bool', data=high_low_concat).fit()
        ctrl2_beta = m_ctrl2.params['is_high_cog']
        ctrl2_p = m_ctrl2.pvalues['is_high_cog']
    except:
        ctrl2_beta = ctrl2_p = np.nan
    
    log(f"    Diff (high - low): {diff:+.6f} [95% CI: {boot_ci_low:.6f}, {boot_ci_high:.6f}]")
    log(f"    t={t_stat:.2f}, p={p_val:.6f} {sig_txt}, d={cohens_d:.3f}")
    
    if p_val < 0.05 and diff < 0:
        log(f"    → ✓ HIGH cognitive load → NEGATIVE deficit (FATIGUE)")
    elif p_val < 0.05 and diff > 0:
        log(f"    → ✗ HIGH cognitive load → POSITIVE deficit (COMPENSATION)")
    else:
        log(f"    → — No significant difference (p={p_val:.4f})")
    
    return {
        'p25': float(p25),
        'p75': float(p75),
        'n_high': len(high),
        'n_low': len(low),
        'n_middle': len(middle),
        'mean_high': float(mean_high), 'se_high': float(se_high),
        'ci_high_low': float(ci_high_low), 'ci_high_high': float(ci_high_high),
        'mean_low': float(mean_low), 'se_low': float(se_low),
        'ci_low_low': float(ci_low_low), 'ci_low_high': float(ci_low_high),
        'diff': float(diff), 'se_diff': float(se_diff),
        'ci_diff_low': float(ci_diff_low), 'ci_diff_high': float(ci_diff_high),
        'boot_ci_low': float(boot_ci_low), 'boot_ci_high': float(boot_ci_high),
        't_stat': float(t_stat), 'p_value': float(p_val), 'sig': sig_txt,
        'cohens_d': float(cohens_d),
        'ctrl_beta': float(ctrl_beta), 'ctrl_p': float(ctrl_p), 'ctrl_sig': ctrl_sig,
        'ctrl_ci_low': float(ctrl_ci_low), 'ctrl_ci_high': float(ctrl_ci_high),
        'ctrl2_beta': float(ctrl2_beta), 'ctrl2_p': float(ctrl2_p),
    }

pct_results = {}

if def_result_clean:
    pct_clean = run_percentile_test(
        def_result_clean['model_data'], 
        'rolling_cog_load_mean_z', 'rolling_phys_load_mean_z',
        "Defenders CLEAN (10-min rolling)")
    pct_decay_clean = run_percentile_test(
        def_result_clean['model_data'],
        'rolling_cog_load_decay_z', 'rolling_phys_load_decay_z',
        "Defenders CLEAN (15-min decay)")
    pct_results['clean'] = {'10min': pct_clean, '15min_decay': pct_decay_clean}

if def_result_full:
    pct_full = run_percentile_test(
        def_result_full['model_data'],
        'rolling_cog_load_mean_z', 'rolling_phys_load_mean_z',
        "Defenders FULL (10-min rolling)")
    pct_decay_full = run_percentile_test(
        def_result_full['model_data'],
        'rolling_cog_load_decay_z', 'rolling_phys_load_decay_z',
        "Defenders FULL (15-min decay)")
    pct_results['full'] = {'10min': pct_full, '15min_decay': pct_decay_full}

# ═══════════════════════════════════════════
# 8. COMPARISON: FIRST BLOCKS INCLUDED VS EXCLUDED
# ═══════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 8: Comparison — First blocks included vs excluded")
log("=" * 60)

comparison = {}
for wtype in ['10min', '15min_decay']:
    comp = {}
    for variant in ['clean', 'full']:
        if variant in pct_results and pct_results[variant].get(wtype):
            r = pct_results[variant][wtype]
            comp[variant] = {
                'n_high': r['n_high'],
                'n_low': r['n_low'],
                'mean_high': r['mean_high'],
                'mean_low': r['mean_low'],
                'diff': r['diff'],
                'p': r['p_value'],
                'd': r['cohens_d'],
                'ci_low': r['boot_ci_low'],
                'ci_high': r['boot_ci_high'],
                'sig': r['sig'],
            }
    
    if 'clean' in comp and 'full' in comp:
        c, f = comp['clean'], comp['full']
        change = c['diff'] - f['diff']
        dir_flip = (np.sign(c['diff']) != np.sign(f['diff']))
        
        log(f"\n  {wtype}:")
        log(f"    FULL (contaminated):  diff={f['diff']:+.6f}, p={f['p']:.6f}, d={f['d']:.3f}")
        log(f"    CLEAN (first blocks removed): diff={c['diff']:+.6f}, p={c['p']:.6f}, d={c['d']:.3f}")
        log(f"    Change in diff: {change:+.6f}")
        log(f"    Direction flip: {dir_flip}")
        
        comparison[wtype] = {
            'full': f,
            'clean': c,
            'change': change,
            'direction_flip': dir_flip,
        }

# ═══════════════════════════════════════════
# 9. GENERATE REPORT
# ═══════════════════════════════════════════
log("\n--- Generating report ---")

md = []
md.append("# Clean Percentile Fatigue Analysis")
md.append("## Excluding First Blocks from Percentile Split")
md.append("")
md.append("**Date:** 2026-07-18")
md.append("")
md.append("## Problem Statement")
md.append("")
md.append("The first 1-2 blocks of every game have zero accumulated rolling load (no preceding blocks),")
md.append("which artificially pulls the \"low load\" group toward a baseline of fresh-start performance.")
md.append("This contamination makes the percentile comparison show reversed or null effects.")
md.append("")
md.append(f"**Fix applied:** Excluded first {N_REMOVE_BLOCKS} blocks per player per game where rolling window ")
md.append(f"has no data or minimal data, before computing percentile splits.")
md.append("")

# Methodology
md.append("## Methodology")
md.append("")
md.append("### 1. Position Labels")
md.append("")
md.append("Loaded from existing `player_position_lookup.csv`. Positions: CB, FB, DM, CM/W.")
md.append("")
md.append("### 2. Rolling Cognitive Load (Preceding Blocks Only)")
md.append("")
md.append("- **10-min rolling window:** mean of cognitive load composite from 2 preceding blocks")
md.append("- **15-min exponential decay:** EWMA of ALL preceding blocks, τ=15 min")
md.append("- Cognitive load composite: z-score average of pressure_composite, opponents_nearby_mean,")
md.append("  reorientation_count, transition_count, depth_mean")
md.append("- Physical load z-scored and computed identically")
md.append("")
md.append("### 3. Contamination Removal")
md.append("")
md.append(f"- Blocks with `block_num < {N_REMOVE_BLOCKS}` per (player, game) are flagged as contaminated")
md.append(f"- These correspond to blocks where rolling window has 0 or 1 preceding blocks")
md.append(f"- Clean dataset: all remaining blocks (n ≈ {len(def_clean):,} for defenders)")
md.append("")
md.append("### 4. Defensive Group")
md.append("")
md.append(f"- CB + FB + DM: {n_def} players, {df[df['is_defender']].shape[0]:,} total blocks")
md.append(f"- CM/W excluded: {n_off} players")
md.append("")
md.append("### 5. Demand-Adjusted Model")
md.append("")
md.append("- Predict expected reorientation_rate from CURRENT situation (pressure_composite +")
md.append("  opponents_nearby_mean + depth_mean)")
md.append("- **NO reorientation_count in predictors** (avoids collinearity)")
md.append("- Train on CLEAN low-load blocks (below median rolling_cog_load AFTER removing first blocks)")
md.append("- Compute `fatigue_deficit = actual - expected`")
md.append("- Negative deficit = worse than situationally expected = fatigue signal")
md.append("")

# Contamination stats
md.append("## Contamination Impact")
md.append("")
n_total = len(df_full)
n_removed = len(df_full) - len(df_clean)
pct_removed = n_removed / n_total * 100
md.append(f"- **Total blocks (defenders, after 1st-block NaN removal):** {n_total}")
md.append(f"- **Blocks removed as contaminated:** {n_removed} ({pct_removed:.1f}%)")
md.append(f"- **Clean blocks:** {len(def_clean)}")
md.append(f"- **First-block mean rolling cog load:** {df.loc[df['is_first_block'], PRIMARY_COG_RAW].mean():.4f} (set to NaN)")
md.append(f"- **Second-block mean rolling cog load:** {df.loc[df['block_num']==1, PRIMARY_COG_RAW].dropna().mean():.4f}")
md.append(f"- **All other blocks mean rolling cog load:** {df.loc[df['block_num']>=2, PRIMARY_COG_RAW].dropna().mean():.4f}")
md.append("")

# Demand model
md.append("## Demand Model Quality")
md.append("")
if def_result_clean:
    md.append(f"### Clean Defenders")
    md.append(f"- **R²:** {def_result_clean['r2']:.4f} (variance explained by situational factors)")
    md.append(f"- **Training set:** {def_result_clean['n_train']:,} low-load blocks")
    md.append(f"- **Predictors:** {', '.join(DEMAND_VARS)}")
    m = def_result_clean['ols']
    md.append("")
    md.append("| Predictor | β | p-value |")
    md.append("|----------|---:|--------:|")
    for var in ['Intercept'] + [f'{v}_z' for v in DEMAND_VARS]:
        sig = '***' if m.pvalues[var] < 0.001 else '**' if m.pvalues[var] < 0.01 else '*' if m.pvalues[var] < 0.05 else ''
        md.append(f"| {var} | {m.params[var]:+.4f} | {m.pvalues[var]:.4f} {sig} |")
    md.append("")

# Continuous model results
md.append("## Continuous Model: `fatigue_deficit ~ rolling_cog_load_z + rolling_phys_load_z`")
md.append("")
md.append("Primary analysis for defenders only.")
md.append("")
md.append("### 10-min Rolling Window (2 preceding blocks)")
md.append("")

if 'clean' in cont_results and cont_results['clean'].get('10min'):
    r = cont_results['clean']['10min']
    md.append(f"- **Cognitive load:** β = {r['β_cog']:+.6f} [95% CI: {r['ci_cog_low']:.6f}, {r['ci_cog_high']:.6f}]")
    md.append(f"- **t = {r['t_cog']:.2f}, p = {r['p_cog']:.6f} {r['sig_cog']}")
    md.append(f"- **Physical load:** β = {r['β_phys']:+.6f} [95% CI: {r['ci_phys_low']:.6f}, {r['ci_phys_high']:.6f}]")
    md.append(f"- **t = {r['t_phys']:.2f}, p = {r['p_phys']:.6f} {r['sig_phys']}")
    md.append(f"- **R² = {r['r2']:.4f}, R²_adj = {r['r2_adj']:.4f}")
    md.append(f"- **Direction: {r['direction']}")
    if r['β_cog'] < 0 and r['p_cog'] < 0.05:
        md.append("- ✅ **Fatigue confirmed:** Higher cognitive load → more negative deficit (worse scanning)")
    elif r['β_cog'] > 0 and r['p_cog'] < 0.05:
        md.append("- ⚠️ **Compensation effect:** Higher cognitive load → more positive deficit (more scanning)")
    else:
        md.append("- ❓ **No significant cognitive fatigue effect** in continuous model")
    md.append("")

md.append("### 15-min Exponential Decay Window")
md.append("")
if 'clean' in cont_results and cont_results['clean'].get('15min_decay'):
    r = cont_results['clean']['15min_decay']
    md.append(f"- **Cognitive load:** β = {r['β_cog']:+.6f} [95% CI: {r['ci_cog_low']:.6f}, {r['ci_cog_high']:.6f}]")
    md.append(f"- **t = {r['t_cog']:.2f}, p = {r['p_cog']:.6f} {r['sig_cog']}")
    md.append(f"- **R² = {r['r2']:.4f}")
    md.append(f"- **Direction: {r['direction']}")
    md.append("")

# Mixed model results
if 'clean' in cont_results and cont_results['clean'].get('mixed'):
    r = cont_results['clean']['mixed']
    md.append("### Mixed Model (with player random effects)")
    md.append(f"- **Cognitive load:** β = {r['β_cog']:+.6f}, p≈{r['p_cog']:.6f}")
    md.append(f"- **Physical load:** β = {r['β_phys']:+.6f}, p≈{r['p_phys']:.6f}")
    md.append("")

# Percentile split results
md.append("## Percentile Split: High vs Low Cognitive Load")
md.append("")
md.append("After removing first blocks, computing new 75th/25th percentiles from clean subset only.")
md.append("")
md.append("**Defenders Only — 10-min Rolling Window**")
md.append("")

if 'clean' in pct_results and pct_results['clean'].get('10min'):
    r = pct_results['clean']['10min']
    md.append(f"- **Low load group (n={r['n_low']:,}):** deficit = {r['mean_low']:+.4f} [95% CI: {r['ci_low_low']:.4f}, {r['ci_low_high']:.4f}]")
    md.append(f"- **High load group (n={r['n_high']:,}):** deficit = {r['mean_high']:+.4f} [95% CI: {r['ci_high_low']:.4f}, {r['ci_high_high']:.4f}]")
    md.append(f"- **Difference (high − low):** {r['diff']:+.6f} [95% CI: {r['boot_ci_low']:.6f}, {r['boot_ci_high']:.6f}]")
    md.append(f"- **p = {r['p_value']:.6f} {r['sig']}, Cohen's d = {r['cohens_d']:.3f}")
    md.append(f"- **Controlled for physical load:** β = {r['ctrl_beta']:+.6f} [95% CI: {r['ctrl_ci_low']:.6f}, {r['ctrl_ci_high']:.6f}], p = {r['ctrl_p']:.6f} {r['ctrl_sig']}")
    
    if r['diff'] < 0 and r['p_value'] < 0.05:
        md.append("- ✅ **Fatigue effect:** High cognitive load → more negative deficit (worse scanning)")
        md.append(f"- **Effect magnitude:** {abs(r['diff']):.4f} scans/block difference between high and low load")
    elif r['diff'] > 0 and r['p_value'] < 0.05:
        md.append("- ⚠️ **Reversed effect:** High cognitive load → more positive deficit (compensation/arousal)")
    else:
        md.append("- ❓ **No significant difference in fatigue deficit between high and low load groups**")
    md.append("")

md.append("**Defenders Only — 15-min Exponential Decay**")
md.append("")

if 'clean' in pct_results and pct_results['clean'].get('15min_decay'):
    r = pct_results['clean']['15min_decay']
    md.append(f"- **Low load group (n={r['n_low']:,}):** deficit = {r['mean_low']:+.4f}")
    md.append(f"- **High load group (n={r['n_high']:,}):** deficit = {r['mean_high']:+.4f}")
    md.append(f"- **Difference:** {r['diff']:+.6f} [95% CI: {r['boot_ci_low']:.6f}, {r['boot_ci_high']:.6f}]")
    md.append(f"- **p = {r['p_value']:.6f} {r['sig']}, Cohen's d = {r['cohens_d']:.3f}")
    md.append("")

# Comparison table
md.append("## Comparison: First Blocks Included vs Excluded")
md.append("")
md.append("This table directly compares the percentile results with and without the contaminated first blocks.")
md.append("")
md.append("| Metric | FULL (contaminated) | CLEAN (first blocks removed) | Change |")
md.append("|--------|-------------------:|----------------------------:|------:|")
for wtype in ['10min', '15min_decay']:
    if wtype in comparison:
        c = comparison[wtype]
        flip_marker = " ⚠️ DIRECTION FLIP" if c['direction_flip'] else ""
        md.append(f"| **{wtype}** | | | |")
        md.append(f"| N(high) | {c['full']['n_high']:,} | {c['clean']['n_high']:,} | {c['clean']['n_high'] - c['full']['n_high']:+} |")
        md.append(f"| N(low) | {c['full']['n_low']:,} | {c['clean']['n_low']:,} | {c['clean']['n_low'] - c['full']['n_low']:+} |")
        md.append(f"| Mean deficit (high) | {c['full']['mean_high']:+.4f} | {c['clean']['mean_high']:+.4f} | {c['clean']['mean_high'] - c['full']['mean_high']:+.4f} |")
        md.append(f"| Mean deficit (low) | {c['full']['mean_low']:+.4f} | {c['clean']['mean_low']:+.4f} | {c['clean']['mean_low'] - c['full']['mean_low']:+.4f} |")
        md.append(f"| Difference | {c['full']['diff']:+.6f} | {c['clean']['diff']:+.6f} | {c['change']:+.6f} |")
        md.append(f"| p-value | {c['full']['p']:.6f} | {c['clean']['p']:.6f} | — |")
        md.append(f"| Cohen's d | {c['full']['d']:.3f} | {c['clean']['d']:.3f} | {c['clean']['d'] - c['full']['d']:+.3f} |")
        md.append(f"| 95% CI | [{c['full']['ci_low']:.4f}, {c['full']['ci_high']:.4f}] | [{c['clean']['ci_low']:.4f}, {c['clean']['ci_high']:.4f}] | — |")
        md.append(f"| Direction | {'NEGATIVE' if c['full']['diff'] < 0 else 'POSITIVE'} | {'NEGATIVE' if c['clean']['diff'] < 0 else 'POSITIVE'} | {'FLIP!' if c['direction_flip'] else 'Same'}{flip_marker} |")
        md.append("")

# Summary
md.append("## Summary of Findings")
md.append("")

# Build narrative summary
if 'clean' in cont_results and cont_results['clean'].get('10min'):
    r = cont_results['clean']['10min']
    
    if r['β_cog'] < 0 and r['p_cog'] < 0.05:
        md.append("1. **Continuous model (clean): Cognitive fatigue confirmed.** Higher accumulated cognitive load predicts more negative fatigue deficits (worse-than-expected scanning).")
    elif r['β_cog'] > 0 and r['p_cog'] < 0.05:
        md.append("1. **Continuous model (clean): Compensation effect.** Higher cognitive load predicts more scanning, not less. Possible arousal/compensation mechanism.")
    else:
        md.append("1. **Continuous model (clean): No significant cognitive fatigue effect.** The relationship between cognitive load and fatigue deficit is not significant.")
    
    if r['β_phys'] < 0 and r['p_phys'] < 0.05:
        md.append("2. **Physical load also contributes** to fatigue deficits (β = {:.4f}, p = {:.6f}). Higher physical load predicts worse scanning.".format(r['β_phys'], r['p_phys']))
    elif r['β_phys'] > 0 and r['p_phys'] < 0.05:
        md.append("2. **Physical load shows a positive relationship** (β = {:.4f}, p = {:.6f}). Higher physical load predicts more scanning.".format(r['β_phys'], r['p_phys']))
    else:
        md.append("2. **Physical load not significant** in the continuous model (p = {:.4f}).".format(r['p_phys']))

if 'clean' in pct_results and pct_results['clean'].get('10min'):
    r = pct_results['clean']['10min']
    md.append(f"3. **Percentile split (clean):** The difference between high and low cognitive load blocks in fatigue deficit is {r['diff']:+.4f} (p = {r['p_value']:.6f}, d = {r['cohens_d']:.3f}).")

if comparison.get('10min'):
    c = comparison['10min']
    md.append(f"4. **Effect of removing first blocks:** The difference shifted by {c['change']:+.4f}. The clean version shows {'a more negative' if c['clean']['diff'] < c['full']['diff'] else 'a less negative'} deficit difference compared to the contaminated version.")
    if c['direction_flip']:
        md.append(f"   ⚠️ **Direction flip detected!** The contaminated version shows a {'positive' if c['full']['diff'] > 0 else 'negative'} difference, but the clean version shows {'positive' if c['clean']['diff'] > 0 else 'negative'}.")
    else:
        md.append(f"   Direction is consistent ({'negative' if c['clean']['diff'] < 0 else 'positive'}) in both versions.")

md.append("")
md.append(f"5. **Key methodological insight:** The first {N_REMOVE_BLOCKS} blocks per player per game represent a 'fresh start' baseline that should not be pooled with blocks following accumulated load. Removing these blocks is essential for valid percentile-based fatigue analysis.")

# Write report
report_path = f'{OUT_DIR}/clean_percentile_fatigue.md'
with open(report_path, 'w') as f:
    f.write('\n'.join(md))
log(f"\nReport saved: {report_path}")

# ═══════════════════════════════════════════
# 10. FIGURE
# ═══════════════════════════════════════════
log("\n--- Generating figure ---")

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

# Panel A: Deficit vs Rolling Cognitive Load (continuous, clean defenders)
ax1 = fig.add_subplot(gs[0, 0])
if def_result_clean:
    md_plot = def_result_clean['model_data']
    cog_col = 'rolling_cog_load_mean_z'
    sub = md_plot[[cog_col, 'fatigue_deficit', 'player_id']].dropna().copy()
    
    try:
        sub['decile'] = pd.qcut(sub[cog_col], 10, labels=False, duplicates='drop')
        dmeans = sub.groupby('decile')['fatigue_deficit'].agg(['mean', 'sem', 'count'])
        dcenters = sub.groupby('decile')[cog_col].median().values
        
        ci = dmeans['sem'] * 1.96
        ax1.errorbar(range(len(dcenters)), dmeans['mean'], yerr=ci,
                     fmt='o-', color='#1565C0', capsize=4, capthick=1.5,
                     markersize=8, linewidth=2)
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        # Regression line
        X_line = sm.add_constant(sub[cog_col].values)
        lr = sm.OLS(sub['fatigue_deficit'].values, X_line).fit()
        x_vals = np.linspace(sub[cog_col].min(), sub[cog_col].max(), 100)
        X_pred = sm.add_constant(x_vals)
        y_pred = lr.predict(X_pred)
        ax1.plot(x_vals, y_pred, '--', color='red', alpha=0.5, linewidth=1.5)
        
        ax1.annotate(f'β={lr.params[1]:+.3f}, p={lr.pvalues[1]:.4f}',
                     xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                     ha='left', va='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    except Exception as e:
        ax1.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax1.transAxes)
    
    ax1.set_xlabel('Cognitive Load Decile', fontsize=11)
    ax1.set_ylabel('Fatigue Deficit (scans/block)', fontsize=11)
    ax1.set_title('A) Continuous: Deficit by Cog Load (Clean Defenders)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

# Panel B: High vs Low percentile split (clean defenders)
ax2 = fig.add_subplot(gs[0, 1])
if 'clean' in pct_results and pct_results['clean'].get('10min'):
    r = pct_results['clean']['10min']
    groups = ['Low Cog Load\n(<25th)', 'High Cog Load\n(>75th)']
    means = [r['mean_low'], r['mean_high']]
    errors = [r['se_low'] * 1.96, r['se_high'] * 1.96]
    
    bars = ax2.bar(groups, means, yerr=errors, color=['#81C784', '#E57373'],
                   capsize=5, width=0.5, alpha=0.8)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Significance annotation
    if r['p_value'] < 0.001:
        sig_text = 'p < 0.001'
    elif r['p_value'] < 0.01:
        sig_text = f'p = {r["p_value"]:.3f}'
    else:
        sig_text = f'p = {r["p_value"]:.4f}'
    
    ax2.set_ylabel('Fatigue Deficit (scans/block)', fontsize=11)
    ax2.set_title(f'B) Percentile Split (Clean Defenders)\nDiff={r["diff"]:+.3f}, d={r["cohens_d"]:.2f}, {sig_text}',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add n labels
    ax2.text(0, r['mean_low'] + errors[0] + 0.01, f'n={r["n_low"]:,}', 
             ha='center', fontsize=9)
    ax2.text(1, r['mean_high'] + errors[1] + 0.01, f'n={r["n_high"]:,}',
             ha='center', fontsize=9)

# Panel C: FULL (contaminated) vs CLEAN comparison
ax3 = fig.add_subplot(gs[0, 2])
if comparison.get('10min'):
    c = comparison['10min']
    groups_bar = ['FULL\n(contaminated)', 'CLEAN\n(first blocks removed)']
    diffs = [c['full']['diff'], c['clean']['diff']]
    err_low = [c['full']['diff'] - c['full']['ci_low'], c['clean']['diff'] - c['clean']['ci_low']]
    err_high = [c['full']['ci_high'] - c['full']['diff'], c['clean']['ci_high'] - c['clean']['diff']]
    
    colors = ['#90A4AE', '#1565C0']
    bars = ax3.bar(groups_bar, diffs, yerr=[err_low, err_high], color=colors,
                   capsize=5, width=0.5, alpha=0.8)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    for i, d in enumerate(diffs):
        ax3.text(i, d + max(diffs)*0.1, f'{d:+.4f}', ha='center', fontsize=10, fontweight='bold')
    
    ax3.set_ylabel('Deficit Difference (scans/block)', fontsize=11)
    ax3.set_title('C) FULL vs CLEAN Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

# Panel D: 15-min decay percentile split
ax4 = fig.add_subplot(gs[1, 0])
if 'clean' in pct_results and pct_results['clean'].get('15min_decay'):
    r = pct_results['clean']['15min_decay']
    groups = ['Low Cog Load\n(<25th)', 'High Cog Load\n(>75th)']
    means = [r['mean_low'], r['mean_high']]
    errors = [r['se_low'] * 1.96, r['se_high'] * 1.96]
    
    bars = ax4.bar(groups, means, yerr=errors, color=['#81C784', '#E57373'],
                   capsize=5, width=0.5, alpha=0.8)
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax4.set_ylabel('Fatigue Deficit (scans/block)', fontsize=11)
    ax4.set_title(f'D) Percentile Split (15-min Decay, Clean)\nDiff={r["diff"]:+.3f}, d={r["cohens_d"]:.2f}',
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

# Panel E: Comparison 15-min decay
ax5 = fig.add_subplot(gs[1, 1])
if comparison.get('15min_decay'):
    c = comparison['15min_decay']
    groups_bar = ['FULL', 'CLEAN']
    diffs = [c['full']['diff'], c['clean']['diff']]
    err_low = [c['full']['diff'] - c['full']['ci_low'], c['clean']['diff'] - c['clean']['ci_low']]
    err_high = [c['full']['ci_high'] - c['full']['diff'], c['clean']['ci_high'] - c['clean']['diff']]
    
    bars = ax5.bar(groups_bar, diffs, yerr=[err_low, err_high], 
                   color=['#90A4AE', '#1565C0'],
                   capsize=5, width=0.5, alpha=0.8)
    ax5.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_ylabel('Deficit Difference', fontsize=11)
    ax5.set_title('E) FULL vs CLEAN (15-min Decay)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

# Panel F: Controlled for physical load
ax6 = fig.add_subplot(gs[1, 2])
if 'clean' in pct_results and pct_results['clean'].get('10min'):
    r = pct_results['clean']['10min']
    if not np.isnan(r['ctrl_beta']):
        groups_label = ['Uncontrolled', 'Controlled\n(phys load)']
        vals = [r['diff'], r['ctrl_beta']]
        err = [r['diff'] - r['boot_ci_low'], r['ctrl_beta'] - r['ctrl_ci_low']]
        err_h = [r['boot_ci_high'] - r['diff'], r['ctrl_ci_high'] - r['ctrl_beta']]
        
        bars = ax6.bar(groups_label, vals, yerr=[err, err_h], 
                       color=['#FF8A65', '#4DB6AC'],
                       capsize=5, width=0.5, alpha=0.8)
        ax6.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        sig_note = 'p < 0.05' if r['ctrl_p'] < 0.05 else 'n.s.'
        ax6.text(1, r['ctrl_beta'] + max(err) + 0.02, sig_note, ha='center', fontsize=9)
        
        ax6.set_ylabel('Effect (deficit diff)', fontsize=11)
        ax6.set_title('F) Controlled for Physical Load', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

# Panel G: Deciles scatter
ax7 = fig.add_subplot(gs[2, 0])
if def_result_clean:
    md_plot = def_result_clean['model_data']
    sub = md_plot[['rolling_cog_load_decay_z', 'fatigue_deficit']].dropna()
    ax7.scatter(sub['rolling_cog_load_decay_z'], sub['fatigue_deficit'], 
                alpha=0.05, s=5, color='#2196F3', rasterized=True)
    
    X_line = sm.add_constant(sub['rolling_cog_load_decay_z'].values)
    lr = sm.OLS(sub['fatigue_deficit'].values, X_line).fit()
    x_vals = np.linspace(sub['rolling_cog_load_decay_z'].min(), sub['rolling_cog_load_decay_z'].max(), 100)
    y_pred = lr.predict(sm.add_constant(x_vals))
    ax7.plot(x_vals, y_pred, '-', color='red', linewidth=2)
    ax7.axhline(0, color='gray', linestyle='--', alpha=0.3)
    
    ax7.set_xlabel('Rolling Cognitive Load (z, 15-min decay)', fontsize=11)
    ax7.set_ylabel('Fatigue Deficit', fontsize=11)
    ax7.set_title(f'G) Scatter: Decay Window\nβ={lr.params[1]:+.3f}, p={lr.pvalues[1]:.4f}', 
                  fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.2)

# Panel H: Continuous 10-min scatter
ax8 = fig.add_subplot(gs[2, 1])
if def_result_clean:
    sub = md_plot[['rolling_cog_load_mean_z', 'fatigue_deficit']].dropna()
    ax8.scatter(sub['rolling_cog_load_mean_z'], sub['fatigue_deficit'], 
                alpha=0.05, s=5, color='#1565C0', rasterized=True)
    
    X_line = sm.add_constant(sub['rolling_cog_load_mean_z'].values)
    lr = sm.OLS(sub['fatigue_deficit'].values, X_line).fit()
    x_vals = np.linspace(sub['rolling_cog_load_mean_z'].min(), sub['rolling_cog_load_mean_z'].max(), 100)
    y_pred = lr.predict(sm.add_constant(x_vals))
    ax8.plot(x_vals, y_pred, '-', color='red', linewidth=2)
    ax8.axhline(0, color='gray', linestyle='--', alpha=0.3)
    
    ax8.set_xlabel('Rolling Cognitive Load (z, 10-min rolling)', fontsize=11)
    ax8.set_ylabel('Fatigue Deficit', fontsize=11)
    ax8.set_title(f'H) Scatter: 10-min Rolling\nβ={lr.params[1]:+.3f}, p={lr.pvalues[1]:.4f}',
                  fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.2)

# Panel I: Info panel
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')

info_lines = [
    "CLEAN PERCENTILE FATIGUE ANALYSIS",
    "=" * 30,
    "",
    "Defenders only (CB + FB + DM)",
    f"First {N_REMOVE_BLOCKS} blocks removed per player-game",
    f"Demand model R²: {def_result_clean['r2']:.4f}" if def_result_clean else "Demand model: N/A",
]

if 'clean' in cont_results and cont_results['clean'].get('10min'):
    r = cont_results['clean']['10min']
    info_lines.extend([
        "",
        "Continuous Model (10-min):",
        f"  β(cog) = {r['β_cog']:+.6f}",
        f"  p(cog) = {r['p_cog']:.6f}",
        f"  β(phys)= {r['β_phys']:+.6f}",
        f"  R² = {r['r2']:.4f}",
    ])

if 'clean' in pct_results and pct_results['clean'].get('10min'):
    r = pct_results['clean']['10min']
    info_lines.extend([
        "",
        "Percentile Split (10-min):",
        f"  Δ = {r['diff']:+.4f}",
        f"  p = {r['p_value']:.6f}",
        f"  d = {r['cohens_d']:.3f}",
    ])

if 'full' in pct_results and pct_results['full'].get('10min'):
    r_full = pct_results['full']['10min']
    info_lines.extend([
        "",
        "Percentile Split FULL:",
        f"  Δ = {r_full['diff']:+.4f}",
        f"  p = {r_full['p_value']:.6f}",
    ])

ax9.text(0.05, 0.95, '\n'.join(info_lines), transform=ax9.transAxes,
         fontsize=9, fontfamily='monospace', verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', alpha=0.8))

fig.suptitle('Clean Percentile Fatigue Analysis — Excluding First Blocks from Percentile Split',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
fig_path = f'{OUT_DIR}/clean_percentile_fatigue.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
log(f"Figure saved: {fig_path}")

# ═══════════════════════════════════════════
# 11. SAVE SUMMARY JSON
# ═══════════════════════════════════════════
log("\n--- Saving summary JSON ---")

summary = {
    'n_defenders': int(n_def),
    'n_clean_blocks': len(def_clean),
    'n_full_blocks': len(def_full),
    'n_removed': len(def_full) - len(def_clean),
    'demand_r2_clean': float(def_result_clean['r2']) if def_result_clean else None,
    'demand_r2_full': float(def_result_full['r2']) if def_result_full else None,
    'continuous': {
        'clean': {
            '10min': cont_results.get('clean', {}).get('10min'),
            '15min_decay': cont_results.get('clean', {}).get('15min_decay'),
        },
        'full': {
            '10min': cont_results.get('full', {}).get('10min'),
            '15min_decay': cont_results.get('full', {}).get('15min_decay'),
        },
    },
    'percentile': {
        'clean': {
            '10min': pct_results.get('clean', {}).get('10min'),
            '15min_decay': pct_results.get('clean', {}).get('15min_decay'),
        },
        'full': {
            '10min': pct_results.get('full', {}).get('10min'),
            '15min_decay': pct_results.get('full', {}).get('15min_decay'),
        },
    },
    'comparison': comparison,
}

# Convert numpy values to native Python
def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

summary = convert(summary)

summ_path = f'{OUT_DIR}/clean_percentile_fatigue_summary.json'
with open(summ_path, 'w') as f:
    json.dump(summary, f, indent=2)
log(f"Summary JSON saved: {summ_path}")

log("\n" + "=" * 60)
log("ANALYSIS COMPLETE")
log("=" * 60)
log("\nReport: focus-fatigue/outputs/analysis/clean_percentile_fatigue.md")
log("Figure: focus-fatigue/outputs/analysis/clean_percentile_fatigue.png")
log("Summary: focus-fatigue/outputs/analysis/clean_percentile_fatigue_summary.json")
