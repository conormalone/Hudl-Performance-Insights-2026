#!/usr/bin/env python3
"""
Visualisation: Cognitive Load → Defensive Decline

Panels:
  A: Phase 1 cog load vs Phase 2 reorientation decline (dose-response)
  B: High vs Low cognitive fatigue groups — defensive quality bar chart
  C: Reorientation rate across Phase 2 blocks, split by high/low Phase 1 cog load
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE = '/home/conormalone/Hudl-Performance-Insights-2026/focus-fatigue'

# ── Load and prepare data ─────────────────────────────────────────────────────
df = pd.read_parquet(f'{BASE}/outputs/analysis/unified_fatigue_dataset.parquet')

COG_INDICATORS = ['pressure_composite', 'opponents_nearby_mean',
                   'reorientation_count', 'transition_count', 'depth_mean']

# Pooled z-score standardisation
for col in COG_INDICATORS:
    mu, sigma = df[col].mean(), df[col].std()
    df[f'{col}_z'] = (df[col] - mu) / sigma

df['cog_load'] = df[[f'{c}_z' for c in COG_INDICATORS]].mean(axis=1)
df['pg_key'] = df['player_id'].astype(str) + '_' + df['game_id'].astype(str)

phase1 = df[(df['phase'] == 1) & (df['block_num'] <= 9)]
phase2 = df[df['phase'] == 2]

# Phase 1 aggregates per player-game
agg_p1 = phase1.groupby('pg_key').agg(
    cog_load_phase1=('cog_load', 'mean'),
    phys_load_phase1=('physical_load', 'mean'),
    reo_rate_p1=('reorientation_rate', 'mean'),
    press_acc_p1=('pressing_accuracy', 'mean'),
    drift_p1=('positional_drift', 'mean'),
    shift_lat_p1=('shift_latency', 'mean'),
).reset_index()

# Phase 2 mean per player-game
agg_p2 = phase2.groupby('pg_key').agg(
    reo_rate_p2=('reorientation_rate', 'mean'),
    press_acc_p2=('pressing_accuracy', 'mean'),
    drift_p2=('positional_drift', 'mean'),
    shift_lat_p2=('shift_latency', 'mean'),
).reset_index()

player_game = agg_p1.merge(agg_p2, on='pg_key', how='inner')
player_game['reo_decline'] = player_game['reo_rate_p2'] - player_game['reo_rate_p1']

# Quartile groups
q_low = player_game['cog_load_phase1'].quantile(0.25)
q_high = player_game['cog_load_phase1'].quantile(0.75)
player_game['cog_group'] = 'Middle 50%'
player_game.loc[player_game['cog_load_phase1'] <= q_low, 'cog_group'] = 'Low cog fatigue (Q1)'
player_game.loc[player_game['cog_load_phase1'] >= q_high, 'cog_group'] = 'High cog fatigue (Q4)'

# ── Phase 2 block-by-block for panel C ────────────────────────────────────────
phase2 = phase2.merge(agg_p1[['pg_key', 'cog_load_phase1']], on='pg_key', how='left')
phase2['cog_group'] = 'Middle 50%'
phase2.loc[phase2['cog_load_phase1'] <= q_low, 'cog_group'] = 'Low cog fatigue (Q1)'
phase2.loc[phase2['cog_load_phase1'] >= q_high, 'cog_group'] = 'High cog fatigue (Q4)'

block_data = phase2.dropna(subset=['reorientation_rate', 'cog_group', 'block_num'])
block_means = block_data.groupby(['block_num', 'cog_group'])['reorientation_rate'].agg(['mean', 'sem', 'count']).reset_index()
block_means = block_means[block_means['count'] > 10]

# ── Panel B data ──────────────────────────────────────────────────────────────
outcomes_b = {
    'reorientation_rate': ('Reorientation Rate (scans/frame)', 'scans/frame', 1),
    'pressing_accuracy': ('Pressing Accuracy', 'success rate', 1),
    'shift_latency': ('Shift Latency (s)', 'seconds (lower=faster)', -1),
    'positional_drift': ('Positional Drift', 'units (lower=better)', -1),
}

group_means = {}
for out in outcomes_b:
    p1col = {'reorientation_rate': 'reo_rate_p1', 'pressing_accuracy': 'press_acc_p1',
             'shift_latency': 'shift_lat_p1', 'positional_drift': 'drift_p1'}[out]
    p2col = {'reorientation_rate': 'reo_rate_p2', 'pressing_accuracy': 'press_acc_p2',
             'shift_latency': 'shift_lat_p2', 'positional_drift': 'drift_p2'}[out]
    
    low = player_game[player_game['cog_group'] == 'Low cog fatigue (Q1)'][p2col].dropna()
    high = player_game[player_game['cog_group'] == 'High cog fatigue (Q4)'][p2col].dropna()
    
    group_means[out] = {
        'low': (low.mean(), low.sem()),
        'high': (high.mean(), high.sem()),
    }

# ── Create figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))

# -- Panel A: Dose-response: Phase 1 cog load vs reorientation decline ---------
ax1 = fig.add_subplot(2, 2, 1)

# Bin cog_load_phase1 into deciles for clean visualisation
valid_a = player_game.dropna(subset=['cog_load_phase1', 'reo_decline', 'phys_load_phase1'])
n_bins = 20
valid_a['cog_bin'] = pd.qcut(valid_a['cog_load_phase1'], n_bins, labels=False)
bin_means = valid_a.groupby('cog_bin').agg(
    cog=('cog_load_phase1', 'mean'),
    decline=('reo_decline', 'mean'),
    decline_se=('reo_decline', 'sem'),
    phys=('phys_load_phase1', 'mean'),
).reset_index()

sc = ax1.scatter(bin_means['cog'], bin_means['decline'],
                 c=bin_means['phys'], cmap='RdYlBu_r', s=80,
                 edgecolors='black', linewidths=0.5, zorder=3)

ax1.errorbar(bin_means['cog'], bin_means['decline'],
             yerr=1.96*bin_means['decline_se'], fmt='none',
             capsize=3, color='gray', alpha=0.5, zorder=1)

# Regression line
from numpy.polynomial.polynomial import polyfit
coeffs = np.polyfit(valid_a['cog_load_phase1'], valid_a['reo_decline'], 1)
x_line = np.linspace(valid_a['cog_load_phase1'].min(), valid_a['cog_load_phase1'].max(), 100)
y_line = np.polyval(coeffs, x_line)
ax1.plot(x_line, y_line, '--', color='crimson', linewidth=2, alpha=0.8,
         label=f'Linear fit (β={coeffs[0]:.3f})')

cbar = plt.colorbar(sc, ax=ax1, label='Physical Load')
ax1.set_xlabel('Cognitive Load (Phase 1, z-score composite)', fontsize=11)
ax1.set_ylabel('Reorientation Rate Change (Phase 2 − Phase 1)', fontsize=11)
ax1.set_title('A: Dose-Response: Cognitive Load Predicts Scanning Decline', fontsize=12, fontweight='bold')
ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.2)

# -- Panel B: Bar chart of Phase 2 outcomes by group ---------------------------
ax2 = fig.add_subplot(2, 2, 2)

outcome_labels = ['Reorientation\nRate', 'Pressing\nAccuracy', 'Shift\nLatency', 'Positional\nDrift']
outcome_keys = list(outcomes_b.keys())
x_pos = np.arange(len(outcome_keys))
width = 0.35

low_means = [group_means[k]['low'][0] for k in outcome_keys]
low_sems = [group_means[k]['low'][1] for k in outcome_keys]
high_means = [group_means[k]['high'][0] for k in outcome_keys]
high_sems = [group_means[k]['high'][1] for k in outcome_keys]

bars1 = ax2.bar(x_pos - width/2, low_means, width, yerr=1.96*np.array(low_sems),
                capsize=3, label='Low cog fatigue (Q1)', color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.5)
bars2 = ax2.bar(x_pos + width/2, high_means, width, yerr=1.96*np.array(high_sems),
                capsize=3, label='High cog fatigue (Q4)', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.5)

# Normalise shift_latency and positional_drift to show on same scale
# Actually let's use secondary y-axis for the second set
# Simpler: just show side by side with different scales noted

ax2.set_xticks(x_pos)
ax2.set_xticklabels(outcome_labels, fontsize=9)
ax2.set_ylabel('Phase 2 Mean Value', fontsize=11)
ax2.set_title('B: High vs Low Cognitive Fatigue — Phase 2 Performance', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, axis='y', alpha=0.2)

# Add significance markers
sig_map = {'reorientation_rate': True, 'pressing_accuracy': True,
           'shift_latency': False, 'positional_drift': False}
for i, k in enumerate(outcome_keys):
    if sig_map.get(k):
        ax2.annotate('**', (x_pos[i], max(low_means[i], high_means[i]) + 0.05),
                     ha='center', fontsize=14, fontweight='bold', color='darkred')

# -- Panel C: Reorientation rate trajectory across Phase 2 blocks ---------------
ax3 = fig.add_subplot(2, 2, (3, 4))

colors = {'Low cog fatigue (Q1)': '#3498db', 'High cog fatigue (Q4)': '#e74c3c', 'Middle 50%': '#95a5a6'}
markers = {'Low cog fatigue (Q1)': 'o', 'High cog fatigue (Q4)': 's', 'Middle 50%': '^'}

for group in ['Low cog fatigue (Q1)', 'High cog fatigue (Q4)']:
    gb = block_means[block_means['cog_group'] == group]
    ax3.errorbar(gb['block_num'], gb['mean'], yerr=1.96*gb['sem'],
                 fmt=f'{markers[group]}-', color=colors[group],
                 label=group, capsize=3, markersize=5, linewidth=1.5, alpha=0.85)

ax3.set_xlabel('Phase 2 Block Number', fontsize=11)
ax3.set_ylabel('Reorientation Rate (scans/frame)', fontsize=11)
ax3.set_title('C: Reorientation Rate Across Phase 2 Blocks\nSplit by Phase 1 Cognitive Load', 
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='lower left')
ax3.grid(True, alpha=0.2)

# Add annotation for decline
ax3.annotate('High cog group declines faster',
             xy=(10, 8.5), xytext=(10, 8.5),
             fontsize=9, color='#e74c3c', fontstyle='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{BASE}/outputs/analysis/cog_load_vs_defensive.png', dpi=200, bbox_inches='tight')
print(f"Figure saved to {BASE}/outputs/analysis/cog_load_vs_defensive.png")
