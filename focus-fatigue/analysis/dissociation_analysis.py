#!/usr/bin/env python3
"""
Dissociation Analysis — Cognitive Fatigue vs Physical Load in Football Defenders

Produces a faceted scatter plot showing that cognitive fatigue effects on
defensive quality signals are independent of physical load (dissociation).

Figure layout:
  - X-axis: Cognitive load proxy (reorientation_rate)
  - Y-axis: Signal measure (raw value or deficit from baseline)
  - Columns: Physical load groups (tertiles or binary split of pressure_composite)
  - Rows: One per defensive signal
  - Each panel: scatter + OLS regression line + 95% CI band

Design decisions:
  1. Cognitive load X-axis: reorientation_rate — captures scanning frequency,
     the most direct proxy of cognitive/attentional demand available in the data.
  2. Physical load groups: pressure_composite tertiles (scale-ready); falls back
     gracefully when data is sparse (only 1 match, 1 block currently).
  3. Y-axis: When multiple blocks exist, uses deficit from per-player baseline
     (early blocks). With only 1 block, uses raw signal values.

Usage:
    python3 analysis/dissociation_analysis.py
    python3 analysis/dissociation_analysis.py \\
        --unified outputs/unified_fatigue_dataset.parquet \\
        --output analysis/figures/dissociation_plot
    python3 analysis/dissociation_analysis.py --n-baseline-blocks 3
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from scipy import stats

# ── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNIFIED = PROJECT_ROOT / "outputs" / "unified_fatigue_dataset.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "figures" / "dissociation_plot"

# ── Configuration ───────────────────────────────────────────────────────────

COGNITIVE_LOAD_METRIC = "reorientation_rate"
"""X-axis: reorientation frequency as cognitive/attentional load proxy.

Rationale: Reorientation rate captures how often a player visually scans
their surroundings and reorients body position — a direct cognitive-perceptual
demand that increases under pressure. It measures the player's need to monitor
and re-assess the environment, which is fundamentally a cognitive workload
signal, not a purely physical one.
"""

PHYSICAL_LOAD_METRIC = "pressure_composite"
"""Metric for stratifying physical load. pressure_composite aggregates
opponents_nearby_mean, depth_mean, and reorientation_rate into a single
physical demand score."""

SIGNAL_COLUMNS = [
    "pressing_accuracy",
    "shift_latency",
    "positional_drift",
    "transition_latency",
]

SIGNAL_LABELS = {
    "pressing_accuracy": "Pressing Accuracy",
    "shift_latency": "Shift Latency (s)",
    "positional_drift": "Positional Drift",
    "transition_latency": "Transition Latency (s)",
}

SIGNAL_UNITS = {
    "pressing_accuracy": "accuracy (0–1)",
    "shift_latency": "seconds",
    "positional_drift": "drift",
    "transition_latency": "seconds",
}

N_BASELINE_BLOCKS = 3
"""Early blocks for per-player baseline (default 3 ≈ 15 min match time)."""

# ── Nature-style Aesthetics ─────────────────────────────────────────────────

NATURE_COLORS = {
    "low": "#4C72B0",
    "medium": "#DD8452",
    "high": "#55A868",
    "moderate": "#937860",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ── Data Loading ────────────────────────────────────────────────────────────


def load_unified(path: str) -> pd.DataFrame:
    """Load unified fatigue dataset."""
    df = pd.read_parquet(path)
    for col in ["block_num", "player_id", "game_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Baseline & Deficit Computation ──────────────────────────────────────────


def compute_signal_baselines(
    df: pd.DataFrame,
    n_baseline_blocks: int = N_BASELINE_BLOCKS,
) -> tuple[pd.DataFrame, bool]:
    """Compute per-player per-signal baselines from early-match blocks.

    Returns (baselines_df, used_raw_fallback) where used_raw_fallback=True
    when baseline computation wasn't viable (too few blocks).
    """
    print(f"\n  Computing per-player baselines from first {n_baseline_blocks} blocks...")

    n_blocks = df["block_num"].nunique()
    if n_blocks < 2:
        print(f"    ⚠️  Only {n_blocks} block(s) available. "
              "Cannot compute meaningful within-player baselines. "
              "Will use raw signal values as Y.")
        return pd.DataFrame(), True

    baseline_rows = []
    for signal in SIGNAL_COLUMNS:
        if signal not in df.columns:
            continue
        signal_df = df[df[signal].notna()].copy()
        if len(signal_df) == 0:
            continue

        early = signal_df[signal_df["block_num"] < n_baseline_blocks].copy()
        if len(early) < 2:
            # Not enough early blocks — fall back to all blocks
            print(f"    ⚠️  {signal}: <2 baseline blocks, using all available")
            early = signal_df.copy()
            if len(early) < 2:
                print(f"    ✗ {signal}: insufficient data for baseline")
                continue

        grouped = (
            early.groupby(["game_id", "player_id"])[signal]
            .agg(["mean", "count"])
            .reset_index()
        )
        grouped = grouped.rename(columns={"mean": "baseline", "count": "n_obs"})
        grouped["signal_name"] = signal
        grouped = grouped[grouped["n_obs"] >= 1]
        baseline_rows.append(grouped)

        n_players = grouped["player_id"].nunique()
        print(f"    ✓ {signal}: baselines for {n_players} players")

    if not baseline_rows:
        return pd.DataFrame(), True

    return pd.concat(baseline_rows, ignore_index=True), False


def compute_signal_deficits(
    df: pd.DataFrame,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    """Compute signal deficits (actual - baseline) for each player-block."""
    print("\n  Computing signal deficits (actual - baseline)...")
    result = df.copy()

    for signal in SIGNAL_COLUMNS:
        if signal not in result.columns:
            continue
        bl = baselines[baselines["signal_name"] == signal][
            ["game_id", "player_id", "baseline"]
        ].copy()
        bl = bl.rename(columns={"baseline": f"{signal}_baseline"})
        bl = bl.drop_duplicates(subset=["game_id", "player_id"])

        result = result.merge(bl, on=["game_id", "player_id"], how="left")

        deficit_col = f"{signal}_deficit"
        result[deficit_col] = result[signal] - result[f"{signal}_baseline"]

        n = result[deficit_col].notna().sum()
        print(f"    ✓ {signal}: {n} observations")

    return result


# ── Physical Load Groups ────────────────────────────────────────────────────


def assign_physical_load_groups(
    df: pd.DataFrame,
    metric: str = PHYSICAL_LOAD_METRIC,
) -> pd.DataFrame:
    """Assign physical load groups (tertiles or binary split).

    Fallback logic:
    - ≥3 unique values → proper tertiles (low/medium/high)
    - 2 unique values → binary split (low/high)
    - 1 unique value → single group (moderate)
    """
    print(f"\n  Assigning physical load groups from '{metric}'...")
    result = df.copy()
    valid = result[metric].dropna()
    n_unique = valid.nunique()

    if n_unique >= 3:
        p33 = valid.quantile(0.33)
        p67 = valid.quantile(0.67)
        result["load_group"] = pd.cut(
            result[metric],
            bins=[-np.inf, p33, p67, np.inf],
            labels=["low", "medium", "high"],
        )
        kind = "tertiles"

    elif n_unique == 2:
        # Binary split by unique values
        uniq_vals = sorted(valid.unique())
        threshold = uniq_vals[0] + (uniq_vals[1] - uniq_vals[0]) / 2
        result["load_group"] = np.where(
            result[metric] <= threshold, "low", "high"
        )
        kind = f"binary split ({uniq_vals[0]}/{uniq_vals[1]})"

    else:
        # Single value
        result["load_group"] = "moderate"
        kind = "single group"

    counts = result["load_group"].value_counts()
    print(f"    ✓ Method: {kind}")
    for label in ["low", "moderate", "medium", "high"]:
        if label in counts.index:
            print(f"      {label}: {counts[label]}")

    return result


# ── Plotting ────────────────────────────────────────────────────────────────


def _ols_stats(x, y):
    """Compute OLS regression. Returns dict or None."""
    if len(x) < 3:
        return None
    # Flat-line check: if Y has near-zero variance, skip
    if np.std(y) < 1e-10:
        return None
    try:
        slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
        n = len(x)
        x_mean = x.mean()
        sxx = np.sum((x - x_mean) ** 2)

        return {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_val,
            "p_value": p_val,
            "std_err": std_err,
            "n": n,
            "x_range": (x.min(), x.max()),
            "x_mean": x_mean,
            "sxx": sxx,
        }
    except Exception:
        return None


def _regression_line(ax, ols, color, x_jittered):
    """Plot regression line with confidence band."""
    if ols is None:
        return
    x_line = np.linspace(ols["x_range"][0], ols["x_range"][1], 100)
    y_pred = ols["intercept"] + ols["slope"] * x_line

    # Confidence band
    n = ols["n"]
    if n > 2 and ols["sxx"] > 0:
        t_crit = stats.t.ppf(0.975, n - 2)
        se_fit = np.sqrt(
            ols["std_err"] ** 2
            * (1 / n + (x_line - ols["x_mean"]) ** 2 / ols["sxx"])
        )
        ax.fill_between(
            x_line, y_pred - t_crit * se_fit, y_pred + t_crit * se_fit,
            alpha=0.12, color=color, zorder=1,
        )

    ax.plot(x_line, y_pred, color=color, linewidth=1.2, zorder=2)

    # Annotation
    r2 = ols["r_value"] ** 2
    p = ols["p_value"]
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.add_artist(AnchoredText(
        f"β={ols['slope']:.3f}\nR²={r2:.2f}{stars}",
        loc="lower left", frameon=True,
        prop={"fontsize": 6, "family": "sans-serif"},
        borderpad=0.3, pad=0.3,
    ))


def plot_dissociation_faceted(
    df: pd.DataFrame,
    output_path: str,
    y_label: str = "Signal Value",
    cognitive_metric: str = COGNITIVE_LOAD_METRIC,
    signal_deficit_cols: list[str] | None = None,
    raw_signal_cols: list[str] | None = None,
) -> None:
    """Create the faceted dissociation scatter plot.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready data.
    output_path : str
        Output file prefix (extension added automatically).
    y_label : str
        Y-axis label suffix.
    cognitive_metric : str
        X-axis column name.
    signal_deficit_cols : list[str] | None
        Deficit column names to plot (multi-block mode).
    raw_signal_cols : list[str] | None
        Raw signal column names (single-block fallback).
    """
    # Determine available Y columns and their labels
    y_cols = []
    y_titles = {}
    use_deficits = False

    if signal_deficit_cols:
        avail = [c for c in signal_deficit_cols
                 if c in df.columns and df[c].notna().sum() > 2]
        if avail:
            y_cols = avail
            for c in avail:
                base = c.replace("_deficit", "")
                y_titles[c] = SIGNAL_LABELS.get(base, base)
            use_deficits = True

    if not y_cols and raw_signal_cols:
        avail = [c for c in raw_signal_cols
                 if c in df.columns and df[c].notna().sum() > 2]
        if avail:
            y_cols = avail
            for c in avail:
                y_titles[c] = SIGNAL_LABELS.get(c, c)

    if not y_cols:
        print("  ⚠️  No signal data available for plotting.")
        return

    # Load groups (physical load strata)
    groups_ordered = ["low", "moderate", "medium", "high"]
    present_groups = [
        g for g in groups_ordered
        if g in df["load_group"].values
    ]
    if not present_groups:
        present_groups = sorted(df["load_group"].dropna().unique())

    n_rows = len(y_cols)
    n_cols = len(present_groups)

    if n_cols == 0:
        print("  ⚠️  No physical load groups available.")
        return

    # ── Create figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols + 0.5, 2.2 * n_rows + 0.5),
        sharex="col",
        sharey="row",
        squeeze=False,
    )

    all_have_significant_slopes = True

    for row_idx, y_col in enumerate(y_cols):
        row_title = y_titles[y_col]
        y_label_text = f"{row_title}\n({y_label})" if y_label else row_title

        for col_idx, group in enumerate(present_groups):
            ax = axes[row_idx, col_idx]

            panel = df[df["load_group"] == group].dropna(
                subset=[cognitive_metric, y_col]
            )

            if len(panel) < 3:
                ax.text(0.5, 0.5, "Insufficient\ndata",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=7, color="grey")
                ax.set_xlabel("")
                ax.set_ylabel("")
                continue

            X = panel[cognitive_metric].values.astype(float)
            Y = panel[y_col].values.astype(float)

            color = NATURE_COLORS.get(group, "#555555")

            # ── Scatter ─────────────────────────────────────────────
            jitter_x = np.random.normal(
                0, 0.03 * X.std() if X.std() > 0 else 0.05, size=len(X)
            )
            ax.scatter(
                X + jitter_x, Y,
                c=color, alpha=0.6, s=20, edgecolors="white",
                linewidth=0.3, zorder=3,
            )

            # ── Regression ──────────────────────────────────────────
            ols = _ols_stats(X, Y)
            if ols:
                _regression_line(ax, ols, color, X + jitter_x)
                if abs(ols["slope"]) < 0.001 and ols["p_value"] > 0.05:
                    all_have_significant_slopes = False

            # ── Zero line (for deficits) ──────────────────────────
            if use_deficits:
                ax.axhline(y=0, color="grey", linestyle="--",
                           linewidth=0.5, alpha=0.4, zorder=0)

            # ── Labels ────────────────────────────────────────────
            if row_idx == n_rows - 1:
                ax.set_xlabel(
                    "Cognitive Load\n(Reorientation Rate, Hz)",
                    fontsize=7,
                )
            if col_idx == 0:
                ax.set_ylabel(y_label_text, fontsize=7)

            # ── Column header ─────────────────────────────────────
            if row_idx == 0:
                label_map = {
                    "low": "Low Physical Load",
                    "medium": "Medium Physical Load",
                    "high": "High Physical Load",
                    "moderate": "Moderate Physical Load",
                }
                ax.set_title(
                    label_map.get(group, group.title()),
                    fontsize=7, fontweight="bold", pad=4,
                )

            ax.tick_params(axis="both", labelsize=6)

    # ── Figure title ────────────────────────────────────────────────
    title = (
        "Dissociation of Cognitive Load Effects from Physical Load\n"
        "in Defensive Performance Signals"
    )
    if use_deficits:
        title += "\n(Y: Deficit from Per-Player Baseline)"
    else:
        title += "\n(Y: Raw Signal Values)"

    fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()

    # ── Save ────────────────────────────────────────────────────────
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt, dpi in [("png", 300), ("svg", 150)]:
        save_path = f"{output_path}.{fmt}"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  ✅ Saved: {save_path}")

    plt.close(fig)

    print(f"\n  Figure: {n_rows} signals × {n_cols} physical load groups")
    print(f"  Y-axis: {'deficit from baseline' if use_deficits else 'raw signal values'}")


# ── Summary Statistics ──────────────────────────────────────────────────────


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print a summary of the dissociation analysis-ready data."""
    print("\n" + "=" * 60)
    print("  DISSOCIATION ANALYSIS — DATA SUMMARY")
    print("=" * 60)

    print(f"\n  Observations: {len(df)}")
    print(f"  Players:     {df['player_id'].nunique()}")
    print(f"  Matches:     {df['game_id'].nunique()}")
    print(f"  Blocks:      {df['block_num'].nunique()} "
          f"({sorted(df['block_num'].unique())})")

    print(f"\n  X: {COGNITIVE_LOAD_METRIC}")
    print(f"     Range: {df[COGNITIVE_LOAD_METRIC].min():.2f} – "
          f"{df[COGNITIVE_LOAD_METRIC].max():.2f}")

    print(f"\n  Physical load ({PHYSICAL_LOAD_METRIC}):")
    print(f"     Range: {df[PHYSICAL_LOAD_METRIC].min():.2f} – "
          f"{df[PHYSICAL_LOAD_METRIC].max():.2f}")

    print(f"\n  Signal data available:")
    for s in SIGNAL_COLUMNS:
        if s in df.columns:
            n = df[s].notna().sum()
            label = SIGNAL_LABELS.get(s, s)
            print(f"    {'✓' if n > 0 else '✗'} {label}: {n} obs")

    if "load_group" in df.columns:
        print(f"\n  Physical load groups:")
        for g, cnt in df["load_group"].value_counts().items():
            print(f"    {g}: {cnt}")


# ── Main Pipeline ───────────────────────────────────────────────────────────


def run_pipeline(
    unified_path: str = str(DEFAULT_UNIFIED),
    output_path: str = str(DEFAULT_OUTPUT),
    n_baseline_blocks: int = N_BASELINE_BLOCKS,
) -> None:
    """Run the full dissociation analysis pipeline."""
    print("=" * 60)
    print("  DISSOCIATION ANALYSIS (SMALL MULTIPLES)")
    print("  Cognitive Fatigue × Physical Load in Defenders")
    print("=" * 60)

    # ── Step 1: Load ──────────────────────────────────────────────────
    print("\n[1/5] Loading unified dataset...")
    df = load_unified(unified_path)
    print(f"  Loaded: {len(df)} rows × {len(df.columns)} columns")

    # ── Step 2: Baselines ────────────────────────────────────────────
    print("\n[2/5] Computing per-player signal baselines...")
    baselines, used_raw = compute_signal_baselines(df, n_baseline_blocks)
    y_label = "Deficit from Baseline"

    if used_raw:
        # Use raw signal values as Y when baseline not viable
        print("\n  → Using raw signal values (no baseline subtraction).")
        for s in SIGNAL_COLUMNS:
            if s in df.columns:
                df[f"{s}_y"] = df[s]
        y_columns = [f"{s}_y" for s in SIGNAL_COLUMNS]
        y_label = "Signal Value"
    else:
        # ── Step 3: Deficits ─────────────────────────────────────────
        print("\n[3/5] Computing signal deficits...")
        df = compute_signal_deficits(df, baselines)
        y_columns = [f"{s}_deficit" for s in SIGNAL_COLUMNS]

    # ── Step 4: Physical load groups ─────────────────────────────────
    print("\n[4/5] Assigning physical load groups...")
    df = assign_physical_load_groups(df)

    # ── Summary ──────────────────────────────────────────────────────
    print_summary_stats(df)

    # ── Step 5: Plot ─────────────────────────────────────────────────
    print("\n[5/5] Generating faceted dissociation plot...")
    if used_raw:
        plot_dissociation_faceted(
            df,
            output_path=output_path,
            y_label=y_label,
            cognitive_metric=COGNITIVE_LOAD_METRIC,
            raw_signal_cols=y_columns,
        )
    else:
        plot_dissociation_faceted(
            df,
            output_path=output_path,
            y_label=y_label,
            cognitive_metric=COGNITIVE_LOAD_METRIC,
            signal_deficit_cols=y_columns,
        )

    print(f"\n  ✅ Pipeline complete. Output: {output_path}.*")


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Dissociation analysis — cognitive fatigue vs physical load"
    )
    parser.add_argument(
        "--unified", type=str, default=str(DEFAULT_UNIFIED),
        help="Path to unified parquet dataset",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help="Output prefix for figures",
    )
    parser.add_argument(
        "--n-baseline-blocks", type=int, default=N_BASELINE_BLOCKS,
        help=f"Early blocks for baseline (default: {N_BASELINE_BLOCKS})",
    )
    args = parser.parse_args()
    run_pipeline(
        unified_path=args.unified,
        output_path=args.output,
        n_baseline_blocks=args.n_baseline_blocks,
    )


if __name__ == "__main__":
    main()
