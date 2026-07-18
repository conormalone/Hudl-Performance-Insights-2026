#!/usr/bin/env python3
"""
Conor's Spec: Defensive Group Percentile Fatigue
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import json

BASE = '/home/conormalone/Hudl-Performance-Insights-2026'
DATA = f'{BASE}/focus-fatigue/outputs/analysis/unified_fatigue_dataset.parquet'
LOOKUP = f'{BASE}/focus-fatigue/outputs/analysis/player_position_lookup.csv'
OUT_FILE = f'{BASE}/focus-fatigue/outputs/analysis/conor_spec_fatigue.md'

# 1. Load data
df = pd.read_parquet(DATA)
lookup = pd.read_csv(LOOKUP)
df = df.merge(lookup, on='player_id', how='left')

# 2. Group defensive positions
defensive_positions = ['CB', 'FB', 'DM']
df['is_defender'] = df['position'].isin(defensive_positions)

print("Data loaded. Shape:", df.shape)
print(f"Defenders: {df['is_defender'].sum()}, Non-defenders: {(~df['is_defender']).sum()}")

# Sort by player, game, block for rolling
df = df.sort_values(['player_id', 'game_id', 'block_num']).reset_index(drop=True)

# 3. Compute rolling cognitive load from PRECEDING blocks (10-min rolling = mean of preceding 2 blocks)
# Composite of pressure_composite, opponents_nearby_mean, reorientation_count, transition_count, depth_mean

# First normalize the components per-player to make them composable
def compute_rolling_within_groups(grp):
    """Compute rolling mean of preceding 2 blocks for cognitive load components."""
    cols = ['pressure_composite', 'opponents_nearby_mean', 'reorientation_count', 'transition_count', 'depth_mean']
    result = grp[cols].copy()
    # Rolling mean of preceding 2 blocks - shift by 1 so we exclude current block, then rolling 2
    for c in cols:
        result[c] = grp[c].shift(1).rolling(2, min_periods=1).mean()
    return result

# Standardize components before compositing
from sklearn.preprocessing import StandardScaler

# Compute z-scores for the whole dataset for each component
scaler = StandardScaler()
component_cols = ['pressure_composite', 'opponents_nearby_mean', 'reorientation_count', 'transition_count', 'depth_mean']
df_scaled = df.copy()
for c in component_cols:
    df_scaled[f'{c}_z'] = scaler.fit_transform(df[[c]])

# Keep original columns for later modeling

# Compute preceding rolling means
for c in component_cols:
    zc = f'{c}_z'
    df_scaled[f'{zc}_preceding_roll'] = df_scaled.groupby(['player_id', 'game_id'])[zc].transform(
        lambda x: x.shift(1).rolling(2, min_periods=1).mean()
    )

# Rolling cognitive load = mean of the 5 z-scored preceding rolling components
preceding_cols = [f'{c}_z_preceding_roll' for c in component_cols]
df_scaled['rolling_cognitive_load'] = df_scaled[preceding_cols].mean(axis=1)

# Also compute rolling physical load (preceding blocks)
df_scaled['physical_load_z'] = scaler.fit_transform(df[['physical_load']])
df_scaled['physical_load_preceding_roll'] = df_scaled.groupby(['player_id', 'game_id'])['physical_load_z'].transform(
    lambda x: x.shift(1).rolling(2, min_periods=1).mean()
)

# Map back
df = df_scaled.copy()

# 4. Exclude first 2 blocks per (player, game)
# Block_num starts at 0, so block 0 and 1 are the first two
df['row_in_game'] = df.groupby(['player_id', 'game_id']).cumcount()
df = df[df['row_in_game'] >= 2].reset_index(drop=True)
print(f"After excluding first 2 blocks per player-game: {len(df)} rows")

# 5. Compute 75th percentile = high load, 25th = low load for rolling_cognitive_load
overall_cog_75 = df['rolling_cognitive_load'].quantile(0.75)
overall_cog_25 = df['rolling_cognitive_load'].quantile(0.25)

df['cog_load_group'] = 'mid'
df.loc[df['rolling_cognitive_load'] <= overall_cog_25, 'cog_load_group'] = 'low'
df.loc[df['rolling_cognitive_load'] >= overall_cog_75, 'cog_load_group'] = 'high'

# 6. Compute fatigue deficit
# Expected reorientation_rate from pressure_composite + opponents_nearby_mean + depth_mean (NO reorientation_count)
# Train on blocks below median rolling cognitive load

median_cog = df['rolling_cognitive_load'].median()
train = df[df['rolling_cognitive_load'] <= median_cog].copy()
test = df.copy()

features_fatigue = ['pressure_composite', 'opponents_nearby_mean', 'depth_mean']
X_train = train[features_fatigue].values
y_train = train['reorientation_rate'].values

model = LinearRegression()
model.fit(X_train, y_train)

df['expected_reo_rate'] = model.predict(test[features_fatigue].values)
df['fatigue_deficit'] = df['reorientation_rate'] - df['expected_reo_rate']

print(f"\nFatigue model coefficients:")
for f, c in zip(features_fatigue, model.coef_):
    print(f"  {f}: {c:.6f}")
print(f"  intercept: {model.intercept_:.6f}")
print(f"  R² on training: {model.score(X_train, y_train):.4f}")

# 7. Physical load groups (same percentile method on rolling physical_load)
overall_phys_75 = df['physical_load_preceding_roll'].quantile(0.75)
overall_phys_25 = df['physical_load_preceding_roll'].quantile(0.25)

df['phys_load_group'] = 'mid'
df.loc[df['physical_load_preceding_roll'] <= overall_phys_25, 'phys_load_group'] = 'low'
df.loc[df['physical_load_preceding_roll'] >= overall_phys_75, 'phys_load_group'] = 'high'

# Filter to only high/low for analysis
df_analysis = df[df['cog_load_group'].isin(['high', 'low'])].reset_index(drop=True)

def analyze_group(subset, group_name):
    """Analyze fatigue deficit by load group for a subset of players."""
    sub = subset.copy()
    
    # Defenders analysis
    high = sub[sub['cog_load_group'] == 'high']
    low = sub[sub['cog_load_group'] == 'low']
    
    print(f"\n=== {group_name} ===")
    print(f"High load n: {len(high)}, Low load n: {len(low)}")
    
    high_mean = high['fatigue_deficit'].mean()
    high_se = high['fatigue_deficit'].sem()
    high_ci = 1.96 * high_se
    low_mean = low['fatigue_deficit'].mean()
    low_se = low['fatigue_deficit'].sem()
    low_ci = 1.96 * low_se
    
    diff = low_mean - high_mean  # low load deficit - high load deficit
    diff_se = np.sqrt(high_se**2 + low_se**2)
    diff_ci = 1.96 * diff_se
    
    # t-test
    t_stat, p_val = stats.ttest_ind(high['fatigue_deficit'], low['fatigue_deficit'], equal_var=False)
    
    print(f"High load: mean={high_mean:.4f}, 95% CI [{high_mean-high_ci:.4f}, {high_mean+high_ci:.4f}]")
    print(f"Low load: mean={low_mean:.4f}, 95% CI [{low_mean-low_ci:.4f}, {low_mean+low_ci:.4f}]")
    print(f"Difference (low - high): {diff:.4f}, 95% CI [{diff-diff_ci:.4f}, {diff+diff_ci:.4f}]")
    print(f"t = {t_stat:.4f}, p = {p_val:.6f}")
    
    # Physical load controlled comparison
    # Compare within physical load strata
    phys_results = {}
    for phys_group in ['low', 'high']:
        phys_sub = sub[sub['phys_load_group'] == phys_group]
        if len(phys_sub) < 10:
            phys_results[phys_group] = None
            continue
        high_phys = phys_sub[phys_sub['cog_load_group'] == 'high']
        low_phys = phys_sub[phys_sub['cog_load_group'] == 'low']
        if len(high_phys) < 3 or len(low_phys) < 3:
            phys_results[phys_group] = None
            continue
        t_p, p_p = stats.ttest_ind(high_phys['fatigue_deficit'], low_phys['fatigue_deficit'], equal_var=False)
        phys_results[phys_group] = {
            'high_n': len(high_phys),
            'low_n': len(low_phys),
            'high_mean': high_phys['fatigue_deficit'].mean(),
            'low_mean': low_phys['fatigue_deficit'].mean(),
            'diff': low_phys['fatigue_deficit'].mean() - high_phys['fatigue_deficit'].mean(),
            't': t_p,
            'p': p_p
        }
        print(f"\n  Controlled for physical_load={phys_group}:")
        print(f"    High load mean: {phys_results[phys_group]['high_mean']:.4f}")
        print(f"    Low load mean: {phys_results[phys_group]['low_mean']:.4f}")
        print(f"    Diff: {phys_results[phys_group]['diff']:.4f}")
        print(f"    t={t_p:.4f}, p={p_p:.6f}")
    
    # Also do a 2-way ANOVA-ish: compare high vs low cog within each phys group
    survives_phys = all(
        r is not None and r['p'] < 0.05 
        for r in phys_results.values() if r is not None
    ) if any(r is not None for r in phys_results.values()) else False
    
    return {
        'group': group_name,
        'n_high': len(high),
        'n_low': len(low),
        'high_mean': high_mean,
        'high_ci_lower': high_mean - high_ci,
        'high_ci_upper': high_mean + high_ci,
        'low_mean': low_mean,
        'low_ci_lower': low_mean - low_ci,
        'low_ci_upper': low_mean + low_ci,
        'diff': diff,
        'diff_ci_lower': diff - diff_ci,
        'diff_ci_upper': diff + diff_ci,
        't_stat': t_stat,
        'p_value': p_val,
        'phys_controlled': phys_results,
        'survives_physical_control': survives_phys
    }

# Analyze defenders
defender_subset = df_analysis[df_analysis['is_defender'] == True].reset_index(drop=True)
def_results = analyze_group(defender_subset, "Defenders (CB + FB + DM)")

# Analyze all players
all_results = analyze_group(df_analysis, "All Players")

# 9. Write report
report = f"""# Conor's Spec: Defensive Group Percentile Fatigue

## Methodology

### Position Assignment
Players were clustered into position groups using per-game averages of depth_mean, opponents_nearby_mean, physical_load, and reorientation_rate. Groups: CB, FB, DM, CM/W. Defensive = CB + FB + DM.

### Rolling Cognitive Load
Computed from **preceding blocks only** (10-min rolling = mean of preceding 2 blocks). Composite = mean of z-scored: pressure_composite, opponents_nearby_mean, reorientation_count, transition_count, depth_mean.

### Exclusion
First 2 blocks per (player, game) are excluded — these have no preceding blocks, so rolling load is zero.

### Load Groups
- **High load**: blocks ≥ 75th percentile of rolling cognitive load
- **Low load**: blocks ≤ 25th percentile of rolling cognitive load
- Middle 50% discarded

### Fatigue Deficit
Expected reorientation_rate modeled from pressure_composite + opponents_nearby_mean + depth_mean (NO reorientation_count). Model trained on blocks below median rolling cognitive load. Deficit = actual − expected.

### Physical Load Control
Physical load rolling mean (preceding 2 blocks, z-scored). High/low groups defined by same percentile method. Comparison repeated within each physical load stratum.

---

## Results

### Fatigue Model (Training: blocks below median rolling cognitive load)
- Features: pressure_composite, opponents_nearby_mean, depth_mean
- Coefficients: pressure_composite={model.coef_[0]:.6f}, opponents_nearby_mean={model.coef_[1]:.6f}, depth_mean={model.coef_[2]:.6f}
- Intercept: {model.intercept_:.6f}
- R² on training data: {model.score(X_train, y_train):.4f}
- Training n: {len(train)}

### 1. Defenders (CB + FB + DM)

| Group | n | Mean Deficit | 95% CI |
|-------|---|-------------|--------|
| High cognitive load | {def_results['n_high']} | {def_results['high_mean']:.4f} | [{def_results['high_ci_lower']:.4f}, {def_results['high_ci_upper']:.4f}] |
| Low cognitive load | {def_results['n_low']} | {def_results['low_mean']:.4f} | [{def_results['low_ci_lower']:.4f}, {def_results['low_ci_upper']:.4f}] |

**Difference (low − high):** {def_results['diff']:.4f} [{def_results['diff_ci_lower']:.4f}, {def_results['diff_ci_upper']:.4f}]
**Welch t-test:** t = {def_results['t_stat']:.4f}, p = {def_results['p_value']:.6f}

**Physical load controlled:**
"""
for phys_g in ['low', 'high']:
    r = def_results['phys_controlled'].get(phys_g)
    if r is not None:
        report += f"""
- Within **{phys_g} physical load**: High load mean = {r['high_mean']:.4f} (n={r['high_n']}), Low load mean = {r['low_mean']:.4f} (n={r['low_n']}), Diff = {r['diff']:.4f}, t = {r['t']:.4f}, p = {r['p']:.6f}
"""
    else:
        report += f"\n- Within **{phys_g} physical load**: insufficient data\n"

report += f"""
**Survives physical load control:** {"YES" if def_results['survives_physical_control'] else "NO"}

---

### 2. All Players (no position filter)

| Group | n | Mean Deficit | 95% CI |
|-------|---|-------------|--------|
| High cognitive load | {all_results['n_high']} | {all_results['high_mean']:.4f} | [{all_results['high_ci_lower']:.4f}, {all_results['high_ci_upper']:.4f}] |
| Low cognitive load | {all_results['n_low']} | {all_results['low_mean']:.4f} | [{all_results['low_ci_lower']:.4f}, {all_results['low_ci_upper']:.4f}] |

**Difference (low − high):** {all_results['diff']:.4f} [{all_results['diff_ci_lower']:.4f}, {all_results['diff_ci_upper']:.4f}]
**Welch t-test:** t = {all_results['t_stat']:.4f}, p = {all_results['p_value']:.6f}

**Physical load controlled:**
"""
for phys_g in ['low', 'high']:
    r = all_results['phys_controlled'].get(phys_g)
    if r is not None:
        report += f"""
- Within **{phys_g} physical load**: High load mean = {r['high_mean']:.4f} (n={r['high_n']}), Low load mean = {r['low_mean']:.4f} (n={r['low_n']}), Diff = {r['diff']:.4f}, t = {r['t']:.4f}, p = {r['p']:.6f}
"""
    else:
        report += f"\n- Within **{phys_g} physical load**: insufficient data\n"

report += f"""
**Survives physical load control:** {"YES" if all_results['survives_physical_control'] else "NO"}

---

## Summary

### Defenders (CB + FB + DM)
- Mean fatigue deficit was **{def_results['low_mean']:.4f} at low cognitive load** and **{def_results['high_mean']:.4f} at high cognitive load**
- Difference (low − high): **{def_results['diff']:.4f}** [{def_results['diff_ci_lower']:.4f}, {def_results['diff_ci_upper']:.4f}]
- **p = {def_results['p_value']:.6f}** — {'statistically significant' if def_results['p_value'] < 0.05 else 'NOT statistically significant'} at α = 0.05
- Survives physical load control: **{'YES' if def_results['survives_physical_control'] else 'NO'}**

### All Players
- Mean fatigue deficit was **{all_results['low_mean']:.4f} at low cognitive load** and **{all_results['high_mean']:.4f} at high cognitive load**
- Difference (low − high): **{all_results['diff']:.4f}** [{all_results['diff_ci_lower']:.4f}, {all_results['diff_ci_upper']:.4f}]
- **p = {all_results['p_value']:.6f}** — {'statistically significant' if all_results['p_value'] < 0.05 else 'NOT statistically significant'} at α = 0.05
- Survives physical load control: **{'YES' if all_results['survives_physical_control'] else 'NO'}**
"""

# Save lookup
lookup.to_csv(f'{BASE}/focus-fatigue/outputs/analysis/player_position_lookup.csv', index=False)

# Write output
with open(OUT_FILE, 'w') as f:
    f.write(report)

print(f"\nReport written to {OUT_FILE}")
print(f"\n=== COMPACT SUMMARY ===")
print(f"Defenders: high mean={def_results['high_mean']:.4f}, low mean={def_results['low_mean']:.4f}, diff={def_results['diff']:.4f}, p={def_results['p_value']:.6f}, survives_phys={'YES' if def_results['survives_physical_control'] else 'NO'}")
print(f"All players: high mean={all_results['high_mean']:.4f}, low mean={all_results['low_mean']:.4f}, diff={all_results['diff']:.4f}, p={all_results['p_value']:.6f}, survives_phys={'YES' if all_results['survives_physical_control'] else 'NO'}")
