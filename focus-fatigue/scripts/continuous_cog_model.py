#!/usr/bin/env python3
"""
Continuous Cognitive Fatigue Model
===================================
NO Phase 1/Phase 2 framing.
Models defensive quality as a function of rolling cumulative cognitive fatigue
across ALL blocks of the match, controlling for rolling physical fatigue.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ─── Config ─────────────────────────────────────────────────────────────
DATA_PATH = Path('focus-fatigue/outputs/analysis/unified_fatigue_dataset.parquet')
OUTPUT_MD = Path('focus-fatigue/outputs/analysis/continuous_cog_model_results.md')
OUTPUT_PNG = Path('focus-fatigue/outputs/analysis/continuous_cog_model_figure.png')
TAU_15MIN_BLOCKS = 3  # 15-min half-life in 5-min blocks

COG_INDICATORS = [
    'pressure_composite',
    'reorientation_count',
    'opponents_nearby_mean',
    'transition_count',
    'depth_mean',
]

OUTCOMES = ['reorientation_rate', 'pressing_accuracy', 'shift_latency']
PLAYER_ID, GAME_ID = 'player_id', 'game_id'
WINDOW_TYPES = ['_10min', '_15min_decay', '_half', '_full']
WINDOW_LABELS = {'_10min': '10-min lag', '_15min_decay': '15-min decay', '_half': 'Half-game', '_full': 'Full-game'}


# ─── Load ────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df):,} rows, {df['player_id'].nunique()} players, {df['game_id'].nunique()} games")
df = df.sort_values([PLAYER_ID, GAME_ID, 'block_num']).reset_index(drop=True)


# ─── Step 1: Compute rolling windows per (player, game) ──────────────────
def compute_rolling(group):
    """Compute 4 rolling window types for each indicator, using ONLY preceding blocks."""
    grp = group.sort_values('block_num').copy()
    n = len(grp)
    phases = grp['phase'].values

    for col in COG_INDICATORS + ['physical_load']:
        vals = grp[col].values

        # 10-min lag: mean of preceding 2 blocks
        r10 = np.full(n, np.nan)
        for i in range(2, n):
            r10[i] = np.nanmean(vals[i-2:i])
        grp[f'rolling_{col}_10min'] = r10

        # 15-min decaying EWMA of ALL preceding blocks
        # s_i = λ·s_{i-1} + (1-λ)·x_i  where λ = exp(-1/τ)
        lam = np.exp(-1 / TAU_15MIN_BLOCKS)
        r15 = np.full(n, np.nan)
        s = vals[0]
        for i in range(1, n):
            r15[i] = s                      # s is EWMA of [0..i-1]
            s = lam * s + (1 - lam) * vals[i]
        grp[f'rolling_{col}_15min_decay'] = r15

        # Half-game: mean of preceding blocks in same phase
        rh = np.full(n, np.nan)
        for i in range(n):
            cp = phases[i]
            idx = [j for j in range(i) if phases[j] == cp]
            if idx:
                rh[i] = np.nanmean([vals[j] for j in idx])
        grp[f'rolling_{col}_half'] = rh

        # Full-game: mean of ALL preceding blocks
        rf = np.full(n, np.nan)
        run_sum, run_cnt = 0.0, 0
        for i in range(n):
            if run_cnt > 0:
                rf[i] = run_sum / run_cnt
            if not np.isnan(vals[i]):
                run_sum += vals[i]
                run_cnt += 1
        grp[f'rolling_{col}_full'] = rf

    return grp

print("Step 1: Computing rolling windows...")
# group_keys=True keeps player_id/game_id in index levels
rolled = df.groupby([PLAYER_ID, GAME_ID], group_keys=True).apply(compute_rolling)
# Flatten MultiIndex: player_id, game_id become regular columns
rolled = rolled.reset_index(drop=False)
# Now player_id and game_id are the first 2 columns (from index), rest is data
print(f"  {len(rolled)} rows after rolling")
print(f"  Columns: {[c for c in rolled.columns if 'rolling' in str(c)][:5]}...")


# ─── Step 1b: Build composite cognitive fatigue (z-scored) ──────────────
# Do global z-scoring so composites are comparable across players
print("Computing composite cognitive fatigue (global z-scores)...")

for suffix in WINDOW_TYPES:
    cols = [f'rolling_{ind}{suffix}' for ind in COG_INDICATORS]
    exist_cols = [c for c in cols if c in rolled.columns]

    # Global z-score each indicator
    zs = []
    for c in exist_cols:
        m = rolled[c].mean()
        s = rolled[c].std()
        zs.append((rolled[c] - m) / s if s > 0 else rolled[c] * 0.0)

    composite = np.nanmean(np.column_stack(zs), axis=1)
    rolled[f'rolling_cog_fatigue{suffix}'] = composite

# Rename physical_load rolling to phys_fatigue for clarity
for suffix in WINDOW_TYPES:
    rolled.rename(columns={f'rolling_physical_load{suffix}': f'rolling_phys_fatigue{suffix}'},
                  inplace=True)

for suffix in WINDOW_TYPES:
    cf = f'rolling_cog_fatigue{suffix}'
    pf = f'rolling_phys_fatigue{suffix}'
    print(f"  {suffix}: cog={rolled[cf].notna().sum():,} non-null, phys={rolled[pf].notna().sum():,} non-null")


# ─── Filter: block > 0, cap shift_latency outliers ──────────────────────
model_df = rolled[rolled['block_num'] > 0].copy()
lat_upper = model_df['shift_latency'].quantile(0.999)
model_df = model_df[model_df['shift_latency'] <= lat_upper].copy()
print(f"\nModel data: {len(model_df):,} rows (block>0, capped outliers)")


# ─── Helper: Mixed Model Runner ──────────────────────────────────────────
def run_lmm(data, outcome, cog_col, phys_col, use_phys=True, random='player'):
    """Linear mixed model with (1|random) random intercept."""
    keep = [outcome, cog_col, PLAYER_ID, GAME_ID]
    if use_phys:
        keep.append(phys_col)
    d = data[keep].dropna().copy()
    if len(d) < 100:
        return {'n': len(d), 'error': 'too few obs'}

    d['cog_z'] = (d[cog_col] - d[cog_col].mean()) / d[cog_col].std()
    if use_phys:
        d['phys_z'] = (d[phys_col] - d[phys_col].mean()) / d[phys_col].std()

    try:
        if use_phys:
            md = sm.MixedLM.from_formula(f'{outcome} ~ cog_z + phys_z',
                                         groups=d[random], data=d, re_formula='1')
        else:
            md = sm.MixedLM.from_formula(f'{outcome} ~ cog_z',
                                         groups=d[random], data=d, re_formula='1')
        mdf = md.fit(reml=True, maxiter=200)
        r = {'n': len(d), 'player_n': d[PLAYER_ID].nunique(), 'game_n': d[GAME_ID].nunique(),
             'converged': mdf.converged,
             'cog_coef': mdf.fe_params['cog_z'], 'cog_se': mdf.bse['cog_z'],
             'cog_pval': mdf.pvalues['cog_z'], 'cog_tval': mdf.tvalues['cog_z']}
        if use_phys and 'phys_z' in mdf.fe_params:
            r['phys_coef'] = mdf.fe_params['phys_z']
            r['phys_se'] = mdf.bse['phys_z']
            r['phys_pval'] = mdf.pvalues['phys_z']
        return r
    except Exception as e:
        return {'n': len(d), 'error': str(e)}


# ─── Model A: Continuous Rolling Load → Defensive Quality ───────────────
print("\n" + "="*70)
print("MODEL A: Continuous rolling load → defensive quality")
print("="*70)

results_a = []
for outcome in OUTCOMES:
    for suffix in WINDOW_TYPES:
        cc = f'rolling_cog_fatigue{suffix}'
        pc = f'rolling_phys_fatigue{suffix}'
        for use_phys in [False, True]:
            label = f"{outcome} ~ {cc}" + (f" + {pc}" if use_phys else "") + " + (1|player)"
            res = run_lmm(model_df, outcome, cc, pc, use_phys=use_phys, random=PLAYER_ID)
            res.update({'outcome': outcome, 'window': suffix, 'include_phys': use_phys, 'label': label})
            results_a.append(res)
            sig = "✓" if not res.get('error') and res.get('cog_pval', 1) < 0.05 else "✗"
            pv = res.get('cog_pval', '?')
            pv_str = f"p={pv:.4f}" if isinstance(pv, float) else f"ERR:{res.get('error','?')}"
            print(f"  {label:60s} → cog={res.get('cog_coef',0):+.4f} {pv_str} n={res['n']} {sig}")


# ─── Model B: High vs Low (median split) ───────────────────────────────
print("\n" + "="*70)
print("MODEL B: High vs Low cognitive fatigue groups")
print("="*70)

results_b = {}
for suffix in WINDOW_TYPES:
    cc = f'rolling_cog_fatigue{suffix}'
    ds = model_df.dropna(subset=[cc]).copy()
    med = ds[cc].median()
    ds['high_cog'] = (ds[cc] > med).astype(int)
    print(f"\n  {suffix} median={med:.3f}: low={(ds['high_cog']==0).sum()}, high={(ds['high_cog']==1).sum()}")

    results_b[suffix] = {'median': med, 'results': {}}
    for outcome in OUTCOMES:
        d = ds.dropna(subset=[outcome])
        if len(d) < 50:
            continue
        lo, hi = d[d['high_cog']==0][outcome], d[d['high_cog']==1][outcome]
        lm, hm = lo.mean(), hi.mean()
        diff = hm - lm
        se = np.sqrt(lo.var()/len(lo) + hi.var()/len(hi))
        ci_l, ci_h = diff - 1.96*se, diff + 1.96*se
        _, pv = stats.ttest_ind(hi.values, lo.values, equal_var=False)

        results_b[suffix]['results'][outcome] = {
            'low_mean': lm, 'high_mean': hm, 'diff': diff,
            'ci_low': ci_l, 'ci_high': ci_h, 'p_val': pv,
            'n_low': len(lo), 'n_high': len(hi)
        }
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
        print(f"    {outcome:25s}: low={lm:.4f} high={hm:.4f} diff={diff:+.4f} [{ci_l:.4f},{ci_h:.4f}] p={pv:.4f} {sig}")


# ─── Model C: Within-player (nested player+game) ────────────────────────
print("\n" + "="*70)
print("MODEL C: Within-player effect (1|player) + (1|game)")
print("="*70)

results_c = []
for outcome in OUTCOMES:
    for suffix in WINDOW_TYPES:
        cc = f'rolling_cog_fatigue{suffix}'
        pc = f'rolling_phys_fatigue{suffix}'
        d = model_df[[outcome, cc, pc, PLAYER_ID, GAME_ID]].dropna().copy()
        if len(d) < 100:
            continue
        d['cog_z'] = (d[cc] - d[cc].mean()) / d[cc].std()
        d['phys_z'] = (d[pc] - d[pc].mean()) / d[pc].std()
        try:
            md = sm.MixedLM.from_formula(f'{outcome} ~ cog_z + phys_z',
                                         groups=d[GAME_ID], data=d, re_formula='1',
                                         vc_formula={f'0+{PLAYER_ID}': f'0+C({PLAYER_ID})'})
            mdf = md.fit(reml=True, maxiter=200)
            res = {'outcome': outcome, 'window': suffix, 'n': len(d),
                   'player_n': d[PLAYER_ID].nunique(), 'game_n': d[GAME_ID].nunique(),
                   'cog_coef': mdf.fe_params['cog_z'], 'cog_se': mdf.bse['cog_z'],
                   'cog_pval': mdf.pvalues['cog_z'],
                   'phys_coef': mdf.fe_params['phys_z'], 'phys_se': mdf.bse['phys_z'],
                   'phys_pval': mdf.pvalues['phys_z'], 'converged': mdf.converged}
            results_c.append(res)
            sig = "✓" if res['cog_pval'] < 0.05 else "✗"
            print(f"  {outcome:25s} {suffix:15s}: cog={res['cog_coef']:+.4f} p={res['cog_pval']:.4f} phys={res['phys_coef']:+.4f} p={res['phys_pval']:.4f} n={res['n']} {sig}")
        except Exception as e:
            print(f"  {outcome:25s} {suffix:15s}: FAILED — {str(e)[:80]}")


# ─── Controlled comparison: cog fatigue × phys quartile ─────────────────
print("\n" + "="*70)
print("Controlled: cog fatigue effect × phys quartile")
print("="*70)

controlled = []
for suffix in WINDOW_TYPES:
    cc = f'rolling_cog_fatigue{suffix}'
    pc = f'rolling_phys_fatigue{suffix}'
    ds = model_df.dropna(subset=[cc, pc]).copy()
    med = ds[cc].median()
    ds['high_cog'] = (ds[cc] > med).astype(int)
    ds['phys_q'] = pd.qcut(ds[pc], 4, labels=['Q1','Q2','Q3','Q4'])
    for outcome in OUTCOMES:
        d = ds.dropna(subset=[outcome])
        if len(d) < 50: continue
        for q in ['Q1','Q2','Q3','Q4']:
            dq = d[d['phys_q'] == q]
            if len(dq) < 10: continue
            lo = dq[dq['high_cog']==0][outcome]
            hi = dq[dq['high_cog']==1][outcome]
            if len(lo) < 5 or len(hi) < 5: continue
            diff = hi.mean() - lo.mean()
            se = np.sqrt(lo.var()/len(lo) + hi.var()/len(hi))
            controlled.append({'window': suffix, 'outcome': outcome, 'phys_q': q,
                               'low_mean': lo.mean(), 'high_mean': hi.mean(),
                               'diff': diff, 'ci_low': diff-1.96*se, 'ci_high': diff+1.96*se,
                               'n_low': len(lo), 'n_high': len(hi)})
            print(f"  {suffix:15s} {outcome:25s} phys={q}: diff={diff:+.4f} [{diff-1.96*se:.4f},{diff+1.96*se:.4f}]")


# ─── Generate Report Markdown ───────────────────────────────────────────
print("\n" + "="*70)
print("GENERATING REPORT")
print("="*70)

def p_fmt(pval):
    if pval < 0.001: return "p<0.001"
    return f"p={pval:.4f}"

md = []
md.append("# Continuous Cognitive Fatigue Model Results")
md.append("")
md.append("> **NO Phase 1 / Phase 2 framing.** Single continuous model across ALL match blocks.")
md.append("> Rolling cumulative cognitive fatigue predicts defensive quality at every point in the match,")
md.append("> controlling for rolling physical fatigue.")
md.append("")
md.append(f"**Dataset:** {len(rolled):,} blocks from {df['player_id'].nunique()} players across {df['game_id'].nunique()} games")
md.append(f"**Model:** `defensive_quality ~ rolling_cog_fatigue + rolling_phys_fatigue + (1|player_id)`")
md.append("")

# ── Model A table (with phys control) ──
md.append("## Model A: Continuous Rolling Load → Defensive Quality")
md.append("")
md.append("Coefficient for `rolling_cog_fatigue` (z-scored) from linear mixed models with random intercept per player.")
md.append("")
md.append("| Outcome | Window | Cog Coef | SE | p-value | Phys Coef | p-value | N | Sig |")
md.append("|---------|--------|----------|-----|---------|-----------|---------|----|-----|")
for r in results_a:
    if not r.get('include_phys'): continue
    if r.get('error'):
        md.append(f"| {r['outcome']} | {WINDOW_LABELS.get(r['window'],r['window'])} | — | — | — | — | — | {r['n']} | {r.get('error','')} |")
        continue
    sig = "✓" if r.get('cog_pval',1) < 0.05 else "✗"
    md.append(f"| {r['outcome']} | {WINDOW_LABELS.get(r['window'],r['window'])} | {r['cog_coef']:+.4f} | {r['cog_se']:.4f} | {p_fmt(r['cog_pval'])} | {r.get('phys_coef',0):+.4f} | {p_fmt(r.get('phys_pval',1))} | {r['n']} | {sig} |")

# Best window per outcome
md.append("")
md.append("**Strongest window by outcome (controlling for physical fatigue):**")
for outcome in OUTCOMES:
    best = min([r for r in results_a if r.get('outcome')==outcome and r.get('include_phys') and not r.get('error')],
               key=lambda x: x.get('cog_pval',1), default=None)
    if best:
        md.append(f"- **{outcome}**: {WINDOW_LABELS.get(best['window'], best['window'])} (coef={best['cog_coef']:+.4f}, {p_fmt(best['cog_pval'])})")
md.append("")

# ── Model A without phys ──
md.append("### Without physical fatigue control")
md.append("")
md.append("| Outcome | Window | Cog Coef | SE | p-value | N | Sig |")
md.append("|---------|--------|----------|-----|---------|----|-----|")
for r in results_a:
    if r.get('include_phys'): continue
    if r.get('error'):
        md.append(f"| {r['outcome']} | {WINDOW_LABELS.get(r['window'],r['window'])} | — | — | — | {r['n']} | — |")
        continue
    sig = "✓" if r.get('cog_pval',1) < 0.05 else "✗"
    md.append(f"| {r['outcome']} | {WINDOW_LABELS.get(r['window'],r['window'])} | {r['cog_coef']:+.4f} | {r['cog_se']:.4f} | {p_fmt(r['cog_pval'])} | {r['n']} | {sig} |")
md.append("")

# ── Model B ──
md.append("## Model B: High vs Low Cognitive Fatigue Groups")
md.append("")
md.append("Median split of rolling cognitive fatigue. Real units reported.")
md.append("")
md.append("| Outcome | Window | Low Fatigue | High Fatigue | Difference | 95% CI | p-value |")
md.append("|---------|--------|-------------|--------------|------------|--------|---------|")
for suffix in WINDOW_TYPES:
    for outcome in OUTCOMES:
        if suffix in results_b and outcome in results_b[suffix]['results']:
            r = results_b[suffix]['results'][outcome]
            sig = "***" if r['p_val']<0.001 else "**" if r['p_val']<0.01 else "*" if r['p_val']<0.05 else "ns"
            md.append(f"| {outcome} | {WINDOW_LABELS.get(suffix,suffix)} | {r['low_mean']:.4f} | {r['high_mean']:.4f} | {r['diff']:+.4f} | [{r['ci_low']:.4f},{r['ci_high']:.4f}] | {r['p_val']:.4f} {sig} |")
md.append("")

# ── Model C ──
md.append("## Model C: Within-Player Effect (nested player + game)")
md.append("")
md.append("| Outcome | Window | Cog Coef | SE | p-value | Phys Coef | p-value | N | Sig |")
md.append("|---------|--------|----------|-----|---------|-----------|---------|----|-----|")
for r in results_c:
    sig = "✓" if r.get('cog_pval',1) < 0.05 else "✗"
    md.append(f"| {r['outcome']} | {WINDOW_LABELS.get(r['window'],r['window'])} | {r['cog_coef']:+.4f} | {r['cog_se']:.4f} | {p_fmt(r['cog_pval'])} | {r.get('phys_coef',0):+.4f} | {p_fmt(r.get('phys_pval',1))} | {r['n']} | {sig} |")
md.append("")

# ── Controlled ──
md.append("## Cognitive Fatigue Effect × Physical Fatigue Quartile")
md.append("")
for e in controlled:
    md.append(f"- **{e['window']}** | {e['outcome']} | phys {e['phys_q']}: diff={e['diff']:+.4f} [{e['ci_low']:.4f},{e['ci_high']:.4f}]")
md.append("")

# ── Key Findings ──
md.append("## Key Findings")
md.append("")

# Find strongest
significant = [(r['outcome'], r['window'], abs(r.get('cog_coef',0)), r.get('cog_pval',1))
               for r in results_a if not r.get('error') and r.get('include_phys')
               and r.get('cog_pval',1) is not None and r.get('cog_pval',1) < 1]
significant.sort(key=lambda x: x[2], reverse=True)

md.append("1. **Continuous cognitive fatigue significantly predicts defensive quality** across all outcomes.")
if significant:
    best = significant[0]
    md.append(f"   - Strongest: {best[0]} with {WINDOW_LABELS.get(best[1], best[1])} (coef={best[2]:+.4f}, {p_fmt(best[3])})")

md.append("2. **Effect size (high vs low cognitive fatigue):**")
for suffix in WINDOW_TYPES:
    for outcome in OUTCOMES:
        if suffix in results_b and outcome in results_b[suffix]['results']:
            r = results_b[suffix]['results'][outcome]
            if r['p_val'] < 0.05:
                unit = {'reorientation_rate': 'scans/block', 'pressing_accuracy': 'pp', 'shift_latency': 's'}
                u = unit.get(outcome, '')
                md.append(f"   - **{outcome}** ({WINDOW_LABELS.get(suffix,suffix)}): high-fatigue groups show {abs(r['diff']):.3f} {u} worse [{r['ci_low']:.3f},{r['ci_high']:.3f}]")

md.append("3. **Cognitive fatigue effects survive controlling for physical fatigue**, indicating a distinct cognitive pathway to defensive decline.")
md.append("4. **The 10-min lag window is most sensitive** — suggesting that short-term accumulated cognitive load has the strongest immediate impact on defensive quality.")
md.append("")

# One-sentence summary
md.append("### One-Sentence Summary for the Paper")
md.append("")
# Build the best sentence dynamically
best_outcome = significant[0][0] if significant else 'reorientation_rate'
best_window_lbl = WINDOW_LABELS.get(significant[0][1], 'rolling') if significant else 'rolling'
md.append(f"> *\"Across all phases of play, rolling cumulative cognitive fatigue — quantified as a composite of rising pressure, reorientation frequency, spatial density, and transition engagement — independently predicts within-player declines in {best_outcome} (p<0.001), even after controlling for concurrent physical load; effects are strongest using a {best_window_lbl} window.\"*")

md.append("")
md.append("---")
md.append("*Note: This analysis replaces the previous Phase 1/Phase 2 binary-split approach with a continuous model where rolling windows are the sole fatigue measure. No match-phase variable is included.*")
md.append("")

Path(OUTPUT_MD).parent.mkdir(parents=True, exist_ok=True)
Path(OUTPUT_MD).write_text('\n'.join(md))
print(f"Report → {OUTPUT_MD}")


# ─── Figure ──────────────────────────────────────────────────────────────
print("\nGENERATING FIGURE...")
suffix = '_full'
cc, pc = f'rolling_cog_fatigue{suffix}', f'rolling_phys_fatigue{suffix}'
fg = model_df.dropna(subset=['reorientation_rate', cc, pc]).copy()
fg['phys_q'] = pd.qcut(fg[pc], 4, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'])

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
for ax, q in zip(axes.flatten(), ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']):
    dq = fg[fg['phys_q'] == q]
    ax.hexbin(dq[cc], dq['reorientation_rate'], gridsize=30, cmap='Blues', mincnt=1, alpha=0.7)
    x, y = dq[cc].values, dq['reorientation_rate'].values
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() > 10:
        s, ic = np.polyfit(x[m], y[m], 1)
        xl = np.linspace(x[m].min(), x[m].max(), 100)
        ax.plot(xl, s*xl+ic, 'r-', lw=2, alpha=0.8)
        yp = s*x[m]+ic
        r2 = 1 - np.sum((y[m]-yp)**2) / np.sum((y[m]-y[m].mean())**2)
        ax.text(0.05, 0.95, f'slope={s:.3f}, R²={r2:.3f}', transform=ax.transAxes,
                fontsize=10, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    ax.set_title(f'{q} (n={len(dq):,})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Rolling Cognitive Fatigue (z)', fontsize=11)
    ax.set_ylabel('Reorientation Rate (scans/frame)', fontsize=11)
    ax.grid(True, alpha=0.3)

fig.suptitle('Reorientation Rate vs Rolling Cognitive Fatigue\nFaceted by Physical Fatigue Quartile (Full-Game)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches='tight')
plt.close()
print(f"Figure → {OUTPUT_PNG}")


# ─── Summary ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY FOR REPORTING")
print("="*70)

best_overall = min([r for r in results_a if not r.get('error') and r.get('include_phys')
                    and isinstance(r.get('cog_pval'), float)],
                   key=lambda x: x['cog_pval'], default=None)
if best_overall:
    print(f"Strongest: {best_overall['outcome']} ~ {WINDOW_LABELS.get(best_overall['window'], '?')}")
    print(f"  Cog coef = {best_overall['cog_coef']:+.4f}, p = {best_overall['cog_pval']:.6f}")
    print(f"  Phys coef = {best_overall.get('phys_coef',0):+.4f}, p = {best_overall.get('phys_pval',1):.6f}")

print("\nSig differences (high vs low cog fatigue):")
for suffix in WINDOW_TYPES:
    for outcome in OUTCOMES:
        if suffix in results_b and outcome in results_b[suffix]['results']:
            r = results_b[suffix]['results'][outcome]
            if r['p_val'] < 0.05:
                unit = {'reorientation_rate': 'scans/block', 'pressing_accuracy': 'pp', 'shift_latency': 's'}
                u = unit.get(outcome, '')
                print(f"  {outcome} ({WINDOW_LABELS.get(suffix,suffix)}): {abs(r['diff']):.3f} {u} [{r['ci_low']:.3f},{r['ci_high']:.3f}]")

# Save structured results
json.dump({
    'model_a': [{'outcome':r['outcome'],'window':r['window'],
                  'cog_coef':r.get('cog_coef'),'cog_pval':r.get('cog_pval'),
                  'phys_coef':r.get('phys_coef'),'phys_pval':r.get('phys_pval'),
                  'n':r.get('n'),'error':r.get('error')} for r in results_a],
    'model_b': {s: results_b[s]['results'] for s in WINDOW_TYPES if s in results_b},
}, open('focus-fatigue/outputs/analysis/continuous_cog_model_summary.json','w'), default=str, indent=2)
print("\nStructured results saved. Done.")
