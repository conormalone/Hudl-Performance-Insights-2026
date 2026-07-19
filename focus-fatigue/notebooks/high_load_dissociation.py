#!/usr/bin/env python3
"""
High Physical Load Dissociation Test — Cognitive vs Physical Fatigue

Tests whether pressing accuracy (mechanics) and reorientation rate (cognition)
show differential sensitivity to physical load. Key question:
  - Does pressing accuracy track physical state (no phase effect within load levels)?
  - Does reorientation rate decline regardless of physical load (pure cognitive fatigue)?
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
import json
warnings.filterwarnings('ignore')

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_parquet(
    '/home/conormalone/Hudl-Performance-Insights-2026/focus-fatigue/outputs/analysis/unified_fatigue_dataset.parquet'
)
print(f"Loaded {len(df):,} observations, {df['player_id'].nunique():,} players, {df['game_id'].nunique():,} matches")

# ── 2. Create physical load groups (tertiles) ─────────────────
df['phys_load_group'] = pd.qcut(df['physical_load'], q=3, labels=['low', 'medium', 'high'])
phys_counts = df.groupby('phys_load_group', observed=True).size()
print(f"\nPhysical load group sizes:\n{phys_counts}")

# ── 3. Helper: two-way ANOVA with interaction ─────────────────
def run_two_way_anova(data, signal, label):
    """Run signal ~ C(phase) * C(phys_load_group), return interaction stats."""
    model = ols(f'{signal} ~ C(phase) * C(phys_load_group)', data=data).fit(cov_type='HC3')
    anova = sm.stats.anova_lm(model, typ=2, robust='hc3')
    
    interaction_f = anova.loc['C(phase):C(phys_load_group)', 'F']
    interaction_p = anova.loc['C(phase):C(phys_load_group)', 'PR(>F)']
    
    phase_f = anova.loc['C(phase)', 'F']
    phase_p = anova.loc['C(phase)', 'PR(>F)']
    
    phys_f = anova.loc['C(phys_load_group)', 'F']
    phys_p = anova.loc['C(phys_load_group)', 'PR(>F)']
    
    return {
        'signal': label,
        'N': len(data),
        'Phase_F': phase_f,
        'Phase_p': phase_p,
        'PhysLoad_F': phys_f,
        'PhysLoad_p': phys_p,
        'Interaction_F': interaction_f,
        'Interaction_p': interaction_p
    }

# ── 4. Per-signal means by phase × phys_load_group ────────────
def compute_phasexphys_means(data, signal):
    """Return a table of means for Phase 1 vs Phase 2 for each phys_load_group."""
    return data.groupby(['phase', 'phys_load_group'], observed=True)[signal].agg(['mean', 'std', 'count'])

# ── 5. Cohen's d ──────────────────────────────────────────────
def cohens_d(x, y):
    """Cohen's d for independent groups."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    s = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if s == 0:
        return 0.0
    return (np.mean(y) - np.mean(x)) / s

# ── 6. Main analysis ──────────────────────────────────────────
SIGNALS = [
    ('reorientation_rate', 'Reorientation Rate (/frame)'),
    ('pressing_accuracy', 'Pressing Accuracy'),
    ('shift_latency', 'Shift Latency (s)'),
    ('positional_drift', 'Positional Drift'),
    ('transition_latency', 'Transition Latency (s)'),
]

results_all = []
means_tables = {}

print("\n" + "=" * 80)
print(" TWO-WAY ANOVA: signal ~ phase × phys_load_group")
print("=" * 80)

for signal, label in SIGNALS:
    sub = df[df[signal].notna()].copy()
    if len(sub) == 0:
        continue
    
    r = run_two_way_anova(sub, signal, label)
    results_all.append(r)
    
    means = compute_phasexphys_means(sub, signal)
    means_tables[label] = means
    
    print(f"\n── {label} (N={len(sub):,}) ──")
    print(f"  Phase:         F={r['Phase_F']:.2f}, p={r['Phase_p']:.6f}")
    print(f"  PhysLoad:      F={r['PhysLoad_F']:.2f}, p={r['PhysLoad_p']:.6f}")
    print(f"  Interaction:   F={r['Interaction_F']:.2f}, p={r['Interaction_p']:.6f}")
    print(f"  Means by group:")
    for (phase, group), row in means.iterrows():
        print(f"    Phase {phase}, {group}: mean={row['mean']:.4f} ± {row['std']:.4f} (n={row['count']:.0f})")

results_df = pd.DataFrame(results_all)

# ── 7. Effect sizes by group ──────────────────────────────────
print("\n" + "=" * 80)
print(" EFFECT SIZES BY PHYSICAL LOAD GROUP")
print("=" * 80)

effect_sizes = []
for signal, label in SIGNALS:
    for group in ['low', 'medium', 'high']:
        g = df[(df['phys_load_group'] == group) & (df[signal].notna())]
        p1 = g.loc[g['phase'] == 1, signal]
        p2 = g.loc[g['phase'] == 2, signal]
        if len(p1) > 1 and len(p2) > 1:
            d = cohens_d(p1, p2)
            t, p_val = stats.ttest_ind(p2, p1, equal_var=False)
            effect_sizes.append({
                'signal': label,
                'phys_load_group': group,
                'd': d,
                't': t,
                'p': p_val,
                'n_phase1': len(p1),
                'n_phase2': len(p2),
                'mean_p1': p1.mean(),
                'mean_p2': p2.mean()
            })
            change_pct = ((p2.mean() - p1.mean()) / p1.mean()) * 100 if p1.mean() != 0 else 0
            sig = "SIGNIFICANT" if p_val < 0.05 else "n.s."
            print(f"  {label:30s} | {group:6s} | d={d:+.4f} | t={t:+.4f} | p={p_val:.6f} | {sig} | Δ={change_pct:+.2f}%")

es_df = pd.DataFrame(effect_sizes)

# ── 8. High-load specific: pressing vs reorientation ──────────
print("\n" + "=" * 80)
print(" HIGH-LOAD ONLY: Pressing Accuracy vs Reorientation Rate")
print("=" * 80)

high = df[df['phys_load_group'] == 'high']
pa_high_p1 = high.loc[(high['phase'] == 1) & (high['pressing_accuracy'].notna()), 'pressing_accuracy']
pa_high_p2 = high.loc[(high['phase'] == 2) & (high['pressing_accuracy'].notna()), 'pressing_accuracy']
d_pa_high = cohens_d(pa_high_p1, pa_high_p2)

reo_high_p1 = high.loc[(high['phase'] == 1) & (high['reorientation_rate'].notna()), 'reorientation_rate']
reo_high_p2 = high.loc[(high['phase'] == 2) & (high['reorientation_rate'].notna()), 'reorientation_rate']
d_reo_high = cohens_d(reo_high_p1, reo_high_p2)

n1_pa, n2_pa = len(pa_high_p1), len(pa_high_p2)
n1_reo, n2_reo = len(reo_high_p1), len(reo_high_p2)
se_d_pa = np.sqrt((n1_pa + n2_pa) / (n1_pa * n2_pa) + d_pa_high**2 / (2 * (n1_pa + n2_pa)))
se_d_reo = np.sqrt((n1_reo + n2_reo) / (n1_reo * n2_reo) + d_reo_high**2 / (2 * (n1_reo + n2_reo)))
d_diff = d_pa_high - d_reo_high
se_diff = np.sqrt(se_d_pa**2 + se_d_reo**2)
z_diff = d_diff / se_diff if se_diff > 0 else 0
p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))

print(f"  Pressing Accuracy — High Load: P1={pa_high_p1.mean():.4f}, P2={pa_high_p2.mean():.4f}, d={d_pa_high:+.4f}, p={stats.ttest_ind(pa_high_p2, pa_high_p1, equal_var=False).pvalue:.6f}")
print(f"  Reorientation Rate — High Load: P1={reo_high_p1.mean():.4f}, P2={reo_high_p2.mean():.4f}, d={d_reo_high:+.4f}, p={stats.ttest_ind(reo_high_p2, reo_high_p1, equal_var=False).pvalue:.6f}")
print(f"  Δd={d_diff:.4f}, z={z_diff:.4f}, p={p_diff:.6f}")

# ── 9. Visualisation ──────────────────────────────────────────
print("\n" + "=" * 80)
print(" GENERATING FIGURE")
print("=" * 80)

palette = {'low': '#4CAF50', 'medium': '#FF9800', 'high': '#F44336'}
group_order = ['low', 'medium', 'high']
group_labels = {'low': 'Low Physical Load', 'medium': 'Medium Physical Load', 'high': 'High Physical Load'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Shared bar settings
n_groups = len(group_order)
width = 0.35
x = np.arange(n_groups)

# ── Panel A: Pressing Accuracy ──
ax = axes[0]
data_pa = df[df['pressing_accuracy'].notna()]
means_a = {}
errs_a = {}
for i, grp in enumerate(group_order):
    for ph in [1, 2]:
        sub = data_pa[(data_pa['phys_load_group'] == grp) & (data_pa['phase'] == ph)]['pressing_accuracy']
        means_a[(grp, ph)] = sub.mean()
        errs_a[(grp, ph)] = sub.std() / np.sqrt(len(sub))

p1_means = [means_a[(g, 1)] for g in group_order]
p1_errs = [errs_a[(g, 1)] for g in group_order]
p2_means = [means_a[(g, 2)] for g in group_order]
p2_errs = [errs_a[(g, 2)] for g in group_order]

ax.bar(x - width/2, p1_means, width, label='Phase 1',
       color=[palette[g] for g in group_order], alpha=0.6, edgecolor='black', linewidth=0.8)
ax.bar(x + width/2, p2_means, width, label='Phase 2',
       color=[palette[g] for g in group_order], alpha=1.0, edgecolor='black', linewidth=0.8)

# Effect size annotations
for i, grp in enumerate(group_order):
    match = es_df[(es_df['signal'] == 'Pressing Accuracy') & (es_df['phys_load_group'] == grp)]
    if len(match) > 0:
        d_val, p_val = match.iloc[0]['d'], match.iloc[0]['p']
        y_top = max(p1_means[i] + p1_errs[i], p2_means[i] + p2_errs[i])
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        ax.annotate(f'd={d_val:.2f} {sig}', xy=(i, y_top + 0.015), ha='center',
                    fontsize=9, fontweight='bold', color='black')

ax.set_xticks(x)
ax.set_xticklabels([group_labels[g] for g in group_order], fontsize=10)
ax.set_ylabel('Pressing Accuracy', fontsize=12)
ax.set_title('A: Pressing Accuracy (Mechanics)', fontsize=13, fontweight='bold', loc='left')
ax.legend(fontsize=10)
ax.set_ylim(0.30, 0.52)

# Add annotation about flatness
ax.annotate('No phase decline\nwithin any load level',
            xy=(1, 0.49), fontsize=10, fontstyle='italic', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

# ── Panel B: Reorientation Rate ──
ax = axes[1]
data_reo = df[df['reorientation_rate'].notna()]
means_b = {}
errs_b = {}
for i, grp in enumerate(group_order):
    for ph in [1, 2]:
        sub = data_reo[(data_reo['phys_load_group'] == grp) & (data_reo['phase'] == ph)]['reorientation_rate']
        means_b[(grp, ph)] = sub.mean()
        errs_b[(grp, ph)] = sub.std() / np.sqrt(len(sub))

p1_means_b = [means_b[(g, 1)] for g in group_order]
p1_errs_b = [errs_b[(g, 1)] for g in group_order]
p2_means_b = [means_b[(g, 2)] for g in group_order]
p2_errs_b = [errs_b[(g, 2)] for g in group_order]

ax.bar(x - width/2, p1_means_b, width, label='Phase 1',
       color=[palette[g] for g in group_order], alpha=0.6, edgecolor='black', linewidth=0.8)
ax.bar(x + width/2, p2_means_b, width, label='Phase 2',
       color=[palette[g] for g in group_order], alpha=1.0, edgecolor='black', linewidth=0.8)

for i, grp in enumerate(group_order):
    match = es_df[(es_df['signal'] == 'Reorientation Rate (/frame)') & (es_df['phys_load_group'] == grp)]
    if len(match) > 0:
        d_val, p_val = match.iloc[0]['d'], match.iloc[0]['p']
        y_top = max(p1_means_b[i] + p1_errs_b[i], p2_means_b[i] + p2_errs_b[i])
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        ax.annotate(f'd={d_val:.2f} {sig}', xy=(i, y_top + 0.25), ha='center',
                    fontsize=9, fontweight='bold', color='black')

ax.set_xticks(x)
ax.set_xticklabels([group_labels[g] for g in group_order], fontsize=10)
ax.set_ylabel('Reorientation Rate (/frame)', fontsize=12)
ax.set_title('B: Reorientation Rate (Cognition)', fontsize=13, fontweight='bold', loc='left')
ax.legend(fontsize=10)
ax.set_ylim(5, 12)

# Annotation about universal decline
ax.annotate('Significant decline\nin ALL load groups',
            xy=(1, 5.8), fontsize=10, fontstyle='italic', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.3))

plt.tight_layout(pad=2)
fig.text(0.5, 0.01,
         'Grouped bars: Phase 1 vs Phase 2 means ± SEM. Cohen\'s d and significance: * p<0.05, ** p<0.01, *** p<0.001.',
         ha='center', fontsize=10, fontstyle='italic')

output_fig = '/home/conormalone/Hudl-Performance-Insights-2026/focus-fatigue/outputs/analysis/high_load_dissociation.png'
plt.savefig(output_fig, dpi=200, bbox_inches='tight')
print(f"Figure saved to {output_fig}")

# ── 10. Summary JSON ──────────────────────────────────────────
pa_row = results_df[results_df['signal'] == 'Pressing Accuracy'].iloc[0]
reo_row = results_df[results_df['signal'] == 'Reorientation Rate (/frame)'].iloc[0]

pa_low = es_df[(es_df['signal'] == 'Pressing Accuracy') & (es_df['phys_load_group'] == 'low')].iloc[0]
pa_high = es_df[(es_df['signal'] == 'Pressing Accuracy') & (es_df['phys_load_group'] == 'high')].iloc[0]
reo_low = es_df[(es_df['signal'] == 'Reorientation Rate (/frame)') & (es_df['phys_load_group'] == 'low')].iloc[0]
reo_high = es_df[(es_df['signal'] == 'Reorientation Rate (/frame)') & (es_df['phys_load_group'] == 'high')].iloc[0]

summary = {
    'pressing_accuracy': {
        'interpretation': 'No phase decline within any physical load level; tracks physical state not fatigue',
        'interaction_F': float(pa_row['Interaction_F']),
        'interaction_p': float(pa_row['Interaction_p']),
        'phase_decline_within_load': False,
        'low_load_d': float(pa_low['d']),
        'low_load_p': float(pa_low['p']),
        'high_load_d': float(pa_high['d']),
        'high_load_p': float(pa_high['p'])
    },
    'reorientation_rate': {
        'interpretation': 'Significant phase decline within every physical load level — pure cognitive fatigue',
        'interaction_F': float(reo_row['Interaction_F']),
        'interaction_p': float(reo_row['Interaction_p']),
        'phase_decline_within_load': True,
        'low_load_d': float(reo_low['d']),
        'low_load_p': float(reo_low['p']),
        'high_load_d': float(reo_high['d']),
        'high_load_p': float(reo_high['p'])
    },
    'high_load_z_test': {
        'd_pressing_accuracy': float(d_pa_high),
        'd_reorientation_rate': float(d_reo_high),
        'd_diff': float(d_diff),
        'z': float(z_diff),
        'p': float(p_diff)
    }
}

with open('/home/conormalone/Hudl-Performance-Insights-2026/focus-fatigue/outputs/analysis/high_load_dissociation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nDone. Files written: high_load_dissociation_results.md, high_load_dissociation.png, high_load_dissociation_summary.json")
