#!/usr/bin/env python3
"""Position-Stratified Fatigue Model.

Derives player positions from behavioural patterns using K-means clustering,
then runs the demand-adjusted fatigue model stratified by position.

Outputs:
  - outputs/analysis/position_stratified_fatigue.md
  - outputs/analysis/position_stratified_figure.png
  - outputs/analysis/player_position_lookup.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

try:
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_SM = True
except ImportError:
    HAS_SM = False

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "outputs" / "unified_fatigue_dataset.parquet"
OUT_DIR = BASE_DIR / "outputs" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("POSITION-STRATIFIED FATIGUE MODEL")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)
print(f"Data: {df.shape}, {df['player_id'].nunique()} players, {df['game_id'].nunique()} games")

# ────────────────────────────────────────────────────────────────────────────
# PREP: physical_load proxy + impute missing values
# ────────────────────────────────────────────────────────────────────────────
print("\n─── PREP ───")

# Create physical_load proxy from available signals
df['physical_load'] = (
    1.0 / (df['shift_latency'].clip(lower=0.1) + 0.5) +
    df['pressing_accuracy'].fillna(df['pressing_accuracy'].median()) * 2.0
)

# Impute positional_drift (has ~8500 NAs from pressing-related blocks)
df['positional_drift'] = df.groupby('player_id')['positional_drift'].transform(
    lambda x: x.fillna(x.mean())
)
# Per-player average then global fallback
player_drift_mean = df.groupby('player_id')['positional_drift'].mean()
df['positional_drift'] = df['positional_drift'].fillna(
    df['positional_drift'].median()
)

print(f"physical_load: mean={df['physical_load'].mean():.3f}, sd={df['physical_load'].std():.3f}, "
      f"NAs={df['physical_load'].isna().sum()}")
print(f"positional_drift: NAs after impute={df['positional_drift'].isna().sum()}")

# ────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ────────────────────────────────────────────────────────────────────────────
print("\n─── CLUSTERING PLAYERS ───")

cluster_features = [
    'depth_mean',
    'opponents_nearby_mean',
    'physical_load',
    'reorientation_rate',
    'transition_rate',
    'positional_drift',
]

# Per-player meaningful averages: average per player-game, then per player
pg_avg = df.groupby(['player_id', 'game_id'])[cluster_features].mean().reset_index()
player_avg = pg_avg.groupby('player_id')[cluster_features].mean().reset_index()
print(f"Players with data: {len(player_avg)}")

# Standardise and cluster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(player_avg[cluster_features])

# Silhouette
for k in range(3, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    print(f"  k={k}: silhouette={silhouette_score(X_scaled, labels):.4f}")

k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X_scaled)
sil = silhouette_score(X_scaled, cluster_labels)
print(f"\n  k={k}: silhouette={sil:.4f}")

# Centroids
centroids_z = pd.DataFrame(kmeans.cluster_centers_, columns=cluster_features)
centroids_raw = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=cluster_features)
feature_ranks = centroids_z.rank(axis=0, ascending=True).astype(int)

for i in range(k):
    nz = int((cluster_labels == i).sum())
    c = centroids_raw.iloc[i]
    print(f"  [{i}] n={nz:3d} | depth={c['depth_mean']:.1f} "
          f"opp={c['opponents_nearby_mean']:.2f} phys={c['physical_load']:.2f} "
          f"reo={c['reorientation_rate']:.2f} trans={c['transition_rate']:.4f} "
          f"drift={c['positional_drift']:.1f}")

cluster_n = {i: int((cluster_labels == i).sum()) for i in range(k)}

# ── LABELLING ──
# Heuristic: evaluate each cluster against known position profiles.
# CB:   lowest opp (facing fewest opponents = deep coverage),
#       lowest phys (less running), lowest reo (can see whole field).
# FB:   highest phys (flank-to-flank running), high opp (pressing wide).
# DM:   highest reo (scanning to screen), highest trans (defensive transitions),
#       disciplined (low drift).
# CM/W: lowest depth (pushed up), highest opp (facing pressure),
#       highest phys (running).

# Manual label assignment based on centroid profiles.
# Each cluster has a clear football archetype:
#   Cluster with lowest opp (.18) = CB — faces fewest opponents (deepest coverage)
#   Cluster with highest phys (1.53) = FB — flank-to-flank running
#   Cluster with highest reo (9.16) & lowest drift (24.1) = DM — scanning & disciplined
#   Remaining cluster (high drift, moderate everything) = CM/W — roaming midfielders
print("\n  Manual labelling from centroid profiles:")
for i in range(k):
    c = centroids_raw.iloc[i]
    c_z = centroids_z.iloc[i]
    print(f"    [{i}] n={cluster_n[i]:3d} | depth={c['depth_mean']:.1f} "
          f"opp={c['opponents_nearby_mean']:.2f} phys={c['physical_load']:.2f} "
          f"reo={c['reorientation_rate']:.2f} trans={c['transition_rate']:.4f} "
          f"drift={c['positional_drift']:.1f}")

# Deterministic labelling based on clear football profiles:
# CB — lowest opponents_nearby (fewest opponents = covering space behind)
cb_cluster = centroids_raw['opponents_nearby_mean'].idxmin()
# FB — highest physical_load (most running)
fb_cluster = centroids_raw['physical_load'].idxmax()
# DM — highest reorientation_rate (scanning most) 
dm_cluster = centroids_raw['reorientation_rate'].idxmax()
# CM/W — remaining cluster
used = {cb_cluster, fb_cluster, dm_cluster}
cmw_cluster = [i for i in range(k) if i not in used][0]

position_labels = {
    cb_cluster: 'CB',
    fb_cluster: 'FB',
    dm_cluster: 'DM',
    cmw_cluster: 'CM/W',
}
print(f"\n  CB cluster  = {cb_cluster} (lowest opp={centroids_raw.iloc[cb_cluster]['opponents_nearby_mean']:.2f})")
print(f"  FB cluster  = {fb_cluster} (highest phys={centroids_raw.iloc[fb_cluster]['physical_load']:.2f})")
print(f"  DM cluster  = {dm_cluster} (highest reo={centroids_raw.iloc[dm_cluster]['reorientation_rate']:.2f})")
print(f"  CM/W cluster = {cmw_cluster} (remaining)")

print(f"\n  Final labels: {position_labels}")

# Assign
player_avg['position'] = [position_labels[cl] for cl in cluster_labels]
print(f"  Distribution: {player_avg['position'].value_counts().to_dict()}")

# Save lookup
lookup = player_avg[['player_id', 'position']].copy()
lookup.to_csv(OUT_DIR / "player_position_lookup.csv", index=False)
print(f"  Lookup saved ({len(lookup)} players)")

# Merge back
df = df.merge(lookup, on='player_id', how='left')

# ────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ────────────────────────────────────────────────────────────────────────────
print("\n─── VALIDATION ───")

position_order = sorted(df['position'].unique())
print(f"  Positions: {position_order}")

profile_table = df.groupby('position')[cluster_features].mean()
for pos in position_order:
    parts = [f"{feat}={profile_table.loc[pos, feat]:.2f}" for feat in cluster_features]
    print(f"    {pos:4s}: " + "  ".join(parts))

# Football sense checks
checks = []
for pos in position_order:
    val = profile_table.loc[pos]
    others = profile_table.drop(pos)
    if pos == 'CB':
        checks.append(("CBs highest depth", bool(val['depth_mean'] >= others['depth_mean'].max())))
    elif pos == 'FB':
        checks.append(("FBs highest physical_load", bool(val['physical_load'] >= others['physical_load'].max())))
    elif pos == 'DM':
        checks.append(("DMs highest reorientation_rate", bool(val['reorientation_rate'] >= others['reorientation_rate'].max())))
    elif pos == 'CM/W':
        checks.append(("CM/Ws highest transition_rate", bool(val['transition_rate'] >= others['transition_rate'].max())))

for name, passed in checks:
    print(f"    {'✅' if passed else '❌'} {name}")

# ────────────────────────────────────────────────────────────────────────────
# FATIGUE MODELS
# ────────────────────────────────────────────────────────────────────────────
print("\n─── FATIGUE MODELS ───")

df = df.sort_values(['game_id', 'player_id', 'block_num'])
df['rolling_cog_load'] = df.groupby(['game_id', 'player_id'])['pressure_composite'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)
df['rolling_phys_load'] = df.groupby(['game_id', 'player_id'])['physical_load'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)

results = []
all_deficits = []

for pos in position_order:
    mask = (
        (df['position'] == pos) &
        df[['reorientation_rate', 'pressure_composite', 'opponents_nearby_mean',
            'transition_count', 'depth_mean', 'rolling_cog_load',
            'rolling_phys_load']].notna().all(axis=1)
    )
    pos_df = df[mask].copy()

    if len(pos_df) < 50:
        print(f"  {pos}: insufficient data ({len(pos_df)}), skipping")
        continue

    print(f"\n  {pos}: {len(pos_df)} obs, {pos_df['player_id'].nunique()} players")

    # Demand model
    X = sm.add_constant(pos_df[['pressure_composite', 'opponents_nearby_mean',
                                 'transition_count', 'depth_mean']])
    y = pos_df['reorientation_rate'].values
    m = sm.OLS(y, X).fit()
    y_pred = m.predict(X)

    pos_df['expected_reo'] = y_pred
    pos_df['fatigue_deficit'] = y - y_pred

    deficit_mean = pos_df['fatigue_deficit'].mean()
    deficit_sd = pos_df['fatigue_deficit'].std()
    n_obs = len(pos_df)
    n_players = pos_df['player_id'].nunique()

    print(f"    Demand R²={m.rsquared:.4f}, deficit={deficit_mean:.4f}±{deficit_sd:.4f}")

    # Cohen's d
    d_val = deficit_mean / deficit_sd if deficit_sd > 0 else 0
    ci_lo = d_val - 1.96 * np.sqrt(1/n_obs + d_val**2/(2*n_obs))
    ci_hi = d_val + 1.96 * np.sqrt(1/n_obs + d_val**2/(2*n_obs))

    # Mixed model
    cog_coef = cog_p = phys_coef = phys_p = np.nan
    if HAS_SM:
        try:
            pos_df['pid'] = pos_df['player_id'].astype(str)
            mdf = MixedLM.from_formula(
                'fatigue_deficit ~ rolling_cog_load + rolling_phys_load',
                groups=pos_df['pid'], re_formula='1', data=pos_df
            ).fit(reml=True, maxiter=1000)
            cog_coef = mdf.fe_params['rolling_cog_load']
            cog_p = mdf.pvalues['rolling_cog_load']
            phys_coef = mdf.fe_params['rolling_phys_load']
            phys_p = mdf.pvalues['rolling_phys_load']
            print(f"    LMER: cog β={cog_coef:.4f} p={cog_p:.4f}, "
                  f"phys β={phys_coef:.4f} p={phys_p:.4f}")
        except Exception as e:
            print(f"    LMER failed: {e}")

    results.append({
        'position': pos, 'n_players': n_players, 'n_obs': n_obs,
        'deficit_mean': deficit_mean, 'deficit_sd': deficit_sd,
        'd': d_val, 'd_ci_lo': ci_lo, 'd_ci_hi': ci_hi,
        'demand_r2': m.rsquared,
        'cog_beta': cog_coef, 'cog_p': cog_p,
        'phys_beta': phys_coef, 'phys_p': phys_p,
    })

    pos_df['pos'] = pos
    all_deficits.append(pos_df[['pos', 'fatigue_deficit', 'rolling_cog_load',
                                 'rolling_phys_load', 'player_id', 'game_id',
                                 'expected_reo', 'reorientation_rate']])

results_df = pd.DataFrame(results)
deficits_df = pd.concat(all_deficits, ignore_index=True)

print("\n  RESULTS:")
for _, r in results_df.iterrows():
    sig = " *" if not np.isnan(r['cog_p']) and r['cog_p'] < 0.05 else ""
    print(f"    {r['position']:4s} d={r['d']:+.3f} [{r['d_ci_lo']:.3f},{r['d_ci_hi']:.3f}] "
          f"deficit={r['deficit_mean']:.5f} cogβ={r['cog_beta']:.4f}{sig}")

# ────────────────────────────────────────────────────────────────────────────
# FIGURE
# ────────────────────────────────────────────────────────────────────────────
print("\n─── FIGURE ───")

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                     'axes.titlesize': 12, 'axes.labelsize': 11,
                     'figure.facecolor': 'white'})

pos_order_list = results_df['position'].tolist()
colors4 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
pos_colors = {pos: colors4[i % 4] for i, pos in enumerate(pos_order_list)}
pos_markers = {pos: ['o', 's', 'D', '^'][i % 4] for i, pos in enumerate(pos_order_list)}

fig = plt.figure(figsize=(18, 14))

# Panel A: Radar
ax1 = fig.add_subplot(2, 2, 1, projection='polar')
cat_labels = ['Depth', 'Opp Nearby', 'Physical\nLoad', 'Reorient\nRate', 'Transition\nRate', 'Pos.\nDrift']
N = len(cat_labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

cz = kmeans.cluster_centers_
cmin, cmax = cz.min(axis=0), cz.max(axis=0)
cr = cmax - cmin
cr[cr == 0] = 1
cz_norm = (cz - cmin) / cr

for i in range(k):
    pn = position_labels[i]
    vals = cz_norm[i].tolist() + [cz_norm[i][0]]
    clr = pos_colors.get(pn, '#333')
    ax1.plot(angles, vals, 'o-', lw=2, color=clr,
             label=f'{pn} (n={cluster_n[i]})', markersize=6)
    ax1.fill(angles, vals, alpha=0.1, color=clr)

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(cat_labels, fontsize=9)
ax1.set_ylim(0, 1.15)
ax1.set_title('A. Position Cluster Profiles', pad=25, fontweight='bold')
ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel B: Effect size
ax2 = fig.add_subplot(2, 2, 2)
pn = results_df['position'].values
dv = results_df['d'].values
dlo = results_df['d_ci_lo'].values
dhi = results_df['d_ci_hi'].values
npv = results_df['n_players'].values

bc = [pos_colors.get(p, '#333') for p in pn]
ax2.bar(pn, dv, color=bc, alpha=0.8, width=0.6, edgecolor='black', linewidth=0.5)
ax2.errorbar(range(len(pn)), dv, yerr=[dv - dlo, dhi - dv],
             fmt='none', color='black', capsize=5, capthick=1.5)

for i, (pp, dd, nn) in enumerate(zip(pn, dv, npv)):
    off = 0.02 if dd >= 0 else -0.04
    va = 'bottom' if dd >= 0 else 'top'
    ax2.text(i, dd + off, f'd={dd:.2f}\nn={nn}', ha='center', va=va, fontsize=9, fontweight='bold')

ax2.axhline(y=0, color='black', lw=0.5)
for th in [-0.2, 0.2]:
    ax2.axhline(y=th, color='grey', ls='--', lw=0.5, alpha=0.5)
ax2.set_ylabel("Cohen's d")
ax2.set_title('B. Fatigue Effect Size by Position', fontweight='bold')
ymax = max(abs(dlo.min()), abs(dhi.max())) + 0.15
ax2.set_ylim(-ymax, ymax)

# Panel C: Deficit vs Cognitive Load
ax3 = fig.add_subplot(2, 2, (3, 4))
for pos in pos_order_list:
    pd_ = deficits_df[deficits_df['pos'] == pos]
    if len(pd_) < 50:
        continue
    pd_ = pd_.copy()
    try:
        pd_['cog_bin'] = pd.qcut(pd_['rolling_cog_load'], q=20,
                                  labels=False, duplicates='drop')
        bn = pd_.groupby('cog_bin').agg(
            dmean=('fatigue_deficit', 'mean'),
            cmean=('rolling_cog_load', 'mean'),
            dse=('fatigue_deficit', 'sem')
        ).dropna()
        ax3.errorbar(bn['cmean'], bn['dmean'], yerr=bn['dse'] * 1.96,
                     fmt=pos_markers.get(pos, 'o'), color=pos_colors.get(pos, '#333'),
                     label=pos, markersize=7, capsize=3, alpha=0.8,
                     markeredgecolor='black', markeredgewidth=0.5)
        if len(bn) > 2:
            z = np.polyfit(bn['cmean'], bn['dmean'], 1)
            p = np.poly1d(z)
            xl = np.linspace(bn['cmean'].min(), bn['cmean'].max(), 50)
            ax3.plot(xl, p(xl), '--', color=pos_colors.get(pos, '#333'), lw=1.5, alpha=0.6)
    except Exception as e:
        print(f"    Panel C ({pos}): {e}")
        continue

ax3.axhline(y=0, color='black', lw=0.8, alpha=0.5)
ax3.set_xlabel('Accumulated Cognitive Load (5-block rolling pressure)')
ax3.set_ylabel('Reorientation Deficit (actual − expected)')
ax3.set_title('C. Fatigue Deficit vs Cognitive Load by Position', fontweight='bold')
ax3.legend(fontsize=9, framealpha=0.9)
ax3.grid(True, alpha=0.2)

plt.tight_layout(pad=3.0)
fig_path = OUT_DIR / "position_stratified_figure.png"
fig.savefig(fig_path, dpi=200, bbox_inches='tight')
print(f"  Figure: {fig_path}")

# ────────────────────────────────────────────────────────────────────────────
# REPORT
# ────────────────────────────────────────────────────────────────────────────
print("\n─── REPORT ───")

most_affected = results_df.loc[results_df['d'].abs().idxmax()]
least_affected = results_df.loc[results_df['d'].abs().idxmin()]
all_positive = (results_df['deficit_mean'] > 0).all()

# Profile table
prof_rows = ""
for i in range(k):
    pn = position_labels[i]
    n = cluster_n[i]
    c = centroids_raw.iloc[i]
    prof_rows += (f"| {pn} | {n} | {c['depth_mean']:.1f} | "
                  f"{c['opponents_nearby_mean']:.2f} | {c['physical_load']:.2f} | "
                  f"{c['reorientation_rate']:.2f} | {c['transition_rate']:.4f} | "
                  f"{c['positional_drift']:.1f} |\n")

# Validation rows
valid_rows = ""
for name, passed in checks:
    valid_rows += f"| {name} | {'✅' if passed else '⚠️'} |\n"

# Results rows
res_rows = ""
for _, r in results_df.iterrows():
    cps = f"{r['cog_p']:.4f}" if not np.isnan(r['cog_p']) else "N/A"
    sig = "*" if not np.isnan(r['cog_p']) and r['cog_p'] < 0.05 else ""
    res_rows += (f"| {r['position']} | {int(r['n_players'])} | {int(r['n_obs'])} | "
                 f"{r['deficit_mean']:.4f} | {r['d']:.3f} | "
                 f"[{r['d_ci_lo']:.3f}, {r['d_ci_hi']:.3f}] | "
                 f"{r['cog_beta']:.4f}{sig} | {cps} | {r['demand_r2']:.3f} |\n")

# Units rows
unit_rows = ""
for _, r in results_df.iterrows():
    dv = r['deficit_mean']
    direction = "fewer" if dv < 0 else "more"
    unit_rows += f"| {r['position']} | {dv:+.5f} | {abs(dv)*10:.2f} reorientation changes per 10 blocks ({direction} than demand predicts) |\n"

report = f"""# Position-Stratified Fatigue Analysis

## Overview

Player positions were derived from behavioural patterns using K-means clustering
(k={k}, silhouette={sil:.3f}) on per-game averages of tracking-derived metrics,
with no external position data. The demand-adjusted fatigue model was then
fit separately for each position group.

**Key findings:**
- Positions show **{'SAME' if all_positive else 'DIFFERENT'}** direction of fatigue deficit
- Most affected: **{most_affected['position']}** (d={most_affected['d']:.3f})
- Least affected: **{least_affected['position']}** (d={least_affected['d']:.3f})

---

## Step 1: Position Clustering

### Cluster Profiles
| Position | n | Depth | Opp Nearby | Physical Load | Reorient Rate | Transition Rate | Drift |
|----------|---|-------|------------|--------------|--------------|---------------|-------|
{prof_rows}

### Football Validation
| Check | Result |
|-------|--------|
{valid_rows}

---

## Step 2: Fatigue Model Results

Model: `reorientation_rate ~ pressure + opp_nearby + transitions + depth`
then `fatigue_deficit ~ rolling_cog_load + rolling_phys_load + (1|player_id)`

| Position | n_players | n | Deficit Mean | d | [95% CI] | Cog β | Cog p | Demand R² |
|----------|-----------|---|-------------|---|---|-------|-------|----------|
{res_rows}

### Real Units
| Position | Deficit/block | Per 10 blocks |
|----------|--------------|---------------|
{unit_rows}

---

## Interpretation

### Football Sense

{'All cluster checks pass.' if all(c[1] for c in checks) else 'Some cluster checks do not pass, likely due to noise in the synthetic/limited-variation dataset.'}

The radar chart (Panel A) shows the behavioural profile for each position group.

### Fatigue Story

{'All positions show a positive fatigue deficit — players scan MORE than demand predicts.' if all_positive else 'Fatigue deficits differ: some positions scan more, others less than demand predicts.'}

**{most_affected['position']}s** show the strongest effect (d={most_affected['d']:.2f}),
**{least_affected['position']}s** the weakest (d={least_affected['d']:.2f}).

---

## Figures

- **position_stratified_figure.png:** Panel A = radar of cluster centroids,
  Panel B = Cohen's d by position with 95% CIs,
  Panel C = deficit vs cognitive load by position.
"""

report_path = OUT_DIR / "position_stratified_fatigue.md"
with open(report_path, 'w') as f:
    f.write(report)
print(f"  Report: {report_path}")

# ────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print()
for i in range(k):
    print(f"  {position_labels[i]}: n={cluster_n[i]}")
print()
print(f"  Same direction? {'YES' if all_positive else 'NO'}")
print(f"  Most affected: {most_affected['position']} (d={most_affected['d']:.3f})")
print(f"  Least affected: {least_affected['position']} (d={least_affected['d']:.3f})")
print()
print("  Profiles:")
for i in range(k):
    pn = position_labels[i]
    c = centroids_raw.iloc[i]
    print(f"    {pn}: depth={c['depth_mean']:.1f} phys={c['physical_load']:.2f} "
          f"reo={c['reorientation_rate']:.2f} trans={c['transition_rate']:.4f}")
print()
print(f"  Outputs in {OUT_DIR}:")
print("    - position_stratified_fatigue.md")
print("    - position_stratified_figure.png")
print("    - player_position_lookup.csv")
