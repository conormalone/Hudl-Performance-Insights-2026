#!/usr/bin/env python3
"""
Dissociation Figure — Corrected Framing
========================================
Replaces Phase 1/Phase 2 labels with cognitive load × physical load framing.

Methodology:
  1. Compute each player's cumulative cognitive load (rolling EWMA with τ=15min
     from PRECEDING blocks only) within each player-game sequence.
  2. Within each player, split observations into high/low cognitive load at the
     player's own median (within-player comparison).
  3. Partition the dataset into physical load tertiles (low/medium/high).
  4. Within each physical load tertile, plot mean reorientation rate and
     pressing accuracy for low vs high cognitive load states.

Cognitive load signal: pressure_composite (situational composite of opponent
  proximity, defensive depth, reorientation engagement, and transition rate).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE = '/home/conormalone/Hudl-Performance-Insights-2026/focus-fatigue'
DATA = f'{BASE}/outputs/analysis/unified_fatigue_dataset.parquet'
OUTPUT_PNG = f'{BASE}/outputs/analysis/dissociation_figure_corrected.png'
OUTPUT_PDF = f'{BASE}/outputs/analysis/dissociation_figure_corrected.pdf'

# ─── Config ─────────────────────────────────────────────────────────────
TAU_BLOCKS = 3              # 15-min half-life in 5-min block units
COG_SIGNAL = 'pressure_composite'

# ─── Load data ──────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(DATA)
print(f"  {len(df):,} rows, {df['player_id'].nunique()} players, "
      f"{df['game_id'].nunique()} games")

# Sort for chronological rolling window
df = df.sort_values(['player_id', 'game_id', 'phase', 'block_num']).reset_index(drop=True)
df['block_global'] = df.groupby(['player_id', 'game_id']).cumcount()

# ─── Step 1: Compute rolling cumulative cognitive load ──────────────────
print("Computing rolling cognitive load (EWMA, τ = 3 blocks ≈ 15 min, preceding only)...")

df['cog_load_rolling'] = np.nan
lam = np.exp(-1.0 / TAU_BLOCKS)

groups = list(df.groupby(['player_id', 'game_id']))
for gi, ((pid, gid), gdf) in enumerate(groups):
    if gi % 500 == 0:
        print(f"  Group {gi}/{len(groups)}...")
    gdf = gdf.sort_values('block_global')
    idx = gdf.index.values
    vals = gdf[COG_SIGNAL].values
    n = len(idx)
    if n == 0:
        continue
    df.loc[idx[0], 'cog_load_rolling'] = np.nan
    running = vals[0]
    for i in range(1, n):
        df.loc[idx[i], 'cog_load_rolling'] = running
        running = lam * running + (1 - lam) * vals[i]

print(f"  Non-null rolling cog load: {df['cog_load_rolling'].notna().sum():,}/{len(df):,}")

# ─── Step 2: Within-player median split on rolling cognitive load ───────
print("Within-player median split...")
df['cog_high'] = np.nan
players = df['player_id'].unique()
for pid in players:
    mask = df['player_id'] == pid
    sub = df.loc[mask, 'cog_load_rolling'].dropna()
    if len(sub) < 3:
        continue
    med = sub.median()
    df.loc[mask & (df['cog_load_rolling'] > med), 'cog_high'] = 1
    df.loc[mask & (df['cog_load_rolling'] <= med), 'cog_high'] = 0

print(f"  High cog: {(df['cog_high']==1).sum():,}, "
      f"Low cog: {(df['cog_high']==0).sum():,}")

# ─── Step 3: Physical load tertiles ─────────────────────────────────────
print("Computing physical load tertiles...")
p33, p67 = df['physical_load'].quantile([1/3, 2/3]).values
df['phys_tert'] = pd.cut(df['physical_load'], bins=[-np.inf, p33, p67, np.inf],
                           labels=['Low\nphysical load',
                                   'Medium\nphysical load',
                                   'High\nphysical load'])
TERTILE_ORDER = ['Low\nphysical load', 'Medium\nphysical load', 'High\nphysical load']
print(f"  Boundaries: {p33:.1f}, {p67:.1f}")

# ─── Step 4: Aggregate and test ─────────────────────────────────────────
print("\nAggregating...")
plot_reo = df.dropna(subset=['cog_high', 'phys_tert', 'reorientation_rate']).copy()
plot_press = df.dropna(subset=['cog_high', 'phys_tert', 'pressing_accuracy']).copy()
print(f"  Reorientation: {len(plot_reo):,} blocks")
print(f"  Pressing accuracy: {len(plot_press):,} blocks")

def get_stats(data, outcome):
    results = {}
    for tert in TERTILE_ORDER:
        for cl, cl_label in [(0, 'Low cognitive load'), (1, 'High cognitive load')]:
            sub = data[(data['phys_tert']==tert) & (data['cog_high']==cl)][outcome].dropna()
            if len(sub) < 3:
                results[(tert, cl_label)] = {'mean': np.nan, 'ci': np.nan, 'n': 0}
            else:
                results[(tert, cl_label)] = {
                    'mean': sub.mean(), 'ci': 1.96 * sub.sem(), 'n': len(sub)
                }
    return results

reo_stats = get_stats(plot_reo, 'reorientation_rate')
press_stats = get_stats(plot_press, 'pressing_accuracy')

# ─── Statistical tests ─────────────────────────────────────────────────
print("\nReorientation rate: low cog - high cog difference")
for tert in TERTILE_ORDER:
    l = plot_reo[(plot_reo['phys_tert']==tert)&(plot_reo['cog_high']==0)]['reorientation_rate']
    h = plot_reo[(plot_reo['phys_tert']==tert)&(plot_reo['cog_high']==1)]['reorientation_rate']
    if len(l)>=5 and len(h)>=5:
        t, p = stats.ttest_ind(l, h, equal_var=False)
        print(f"  {tert}: diff={l.mean()-h.mean():+.4f}, t={t:.2f}, p={p:.4f}, "
              f"n_low={len(l):,}, n_high={len(h):,}")

print("\nPressing accuracy: low cog - high cog difference")
for tert in TERTILE_ORDER:
    l = plot_press[(plot_press['phys_tert']==tert)&(plot_press['cog_high']==0)]['pressing_accuracy']
    h = plot_press[(plot_press['phys_tert']==tert)&(plot_press['cog_high']==1)]['pressing_accuracy']
    if len(l)>=5 and len(h)>=5:
        t, p = stats.ttest_ind(l, h, equal_var=False)
        print(f"  {tert}: diff={l.mean()-h.mean():+.4f}, t={t:.2f}, p={p:.4f}, "
              f"n_low={len(l):,}, n_high={len(h):,}")

# ─── Step 5: Create figure ──────────────────────────────────────────────
print("\nCreating figure...")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

COLORS = {
    'Low cognitive load': '#7FB3D8',   # light
    'High cognitive load': '#2E6DA4',  # dark
}

bar_width = 0.32
x = np.arange(len(TERTILE_ORDER))

for idx, (outcome_key, ylabel) in enumerate([
    ('reorientation_rate', 'Movement Reactivity Rate\n(scans / frame)'),
    ('pressing_accuracy', 'Pressing Accuracy\n(success rate)'),
]):
    ax = axes[idx]
    stats_dict = reo_stats if outcome_key == 'reorientation_rate' else press_stats
    data_for_test = plot_reo if outcome_key == 'reorientation_rate' else plot_press

    for j, (cog_label, offset) in enumerate([
        ('Low cognitive load', -bar_width/2),
        ('High cognitive load', bar_width/2)
    ]):
        means = [stats_dict.get((t, cog_label), {}).get('mean', np.nan) for t in TERTILE_ORDER]
        cis = [stats_dict.get((t, cog_label), {}).get('ci', np.nan) for t in TERTILE_ORDER]

        ax.bar(x + offset, means, bar_width,
               color=COLORS[cog_label], alpha=0.85,
               edgecolor='black', linewidth=0.5,
               label=cog_label,
               yerr=cis, capsize=3, error_kw={'linewidth': 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(TERTILE_ORDER, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel('Physical Load Tertile', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    # Significance stars
    for ti, tert in enumerate(TERTILE_ORDER):
        low_v = data_for_test[(data_for_test['phys_tert']==tert)&(data_for_test['cog_high']==0)][outcome_key].dropna()
        high_v = data_for_test[(data_for_test['phys_tert']==tert)&(data_for_test['cog_high']==1)][outcome_key].dropna()
        if len(low_v) >= 5 and len(high_v) >= 5:
            _, p_val = stats.ttest_ind(low_v, high_v, equal_var=False)
            l_mean = low_v.mean()
            h_mean = high_v.mean()
            l_ci = 1.96 * low_v.sem()
            h_ci = 1.96 * high_v.sem()
            top_y = max(l_mean + l_ci, h_mean + h_ci)
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            y_annot = top_y + 0.04 * y_range
            if p_val < 0.001:
                ax.text(ti, y_annot, '***', ha='center', fontsize=11, fontweight='bold')
            elif p_val < 0.01:
                ax.text(ti, y_annot, '**', ha='center', fontsize=11, fontweight='bold')
            elif p_val < 0.05:
                ax.text(ti, y_annot, '*', ha='center', fontsize=11, fontweight='bold')

    if idx == 1:
        ax.legend(fontsize=9, loc='upper right')

    ax.set_xlim(-0.6, len(TERTILE_ORDER) - 0.4)

# Subpanel labels
axes[0].text(-0.18, 1.05, 'A', transform=axes[0].transAxes,
             fontsize=14, fontweight='bold', va='bottom', ha='right')
axes[1].text(-0.18, 1.05, 'B', transform=axes[1].transAxes,
             fontsize=14, fontweight='bold', va='bottom', ha='right')

fig.suptitle('Dissociation of Cognitive Fatigue from Physical Load\n'
             '(Within-Player Cumulative Cognitive Load by Physical Load Tertile)',
             fontsize=13, fontweight='bold', y=1.03)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches='tight')
plt.savefig(OUTPUT_PDF, bbox_inches='tight')
plt.close()

print(f"\nFigure saved:")
print(f"  PNG → {OUTPUT_PNG}")
print(f"  PDF → {OUTPUT_PDF}")
print("Done ✅")
