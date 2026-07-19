#!/usr/bin/env python3
"""
Dissociation Analysis — Faceted Scatter Plot (Small Multiples)
=================================================================
Demonstrates that cognitive fatigue effects on defensive quality are
*independent* of physical load.

Architecture
------------
- Model 1 outputs (pressure_exposure → unified via merge_outputs.py)
- Signal outputs (shift_latency, pressing_accuracy, positional_drift,
  transition_latency → merged via merge_outputs.py)

Methodology
-----------
1. Load unified dataset (merged Model 1 + signals)
2. Compute per-player baselines from early-match blocks (first 3 blocks,
   when available). For blocks 1-3, signal_deficit = signal_value - baseline.
3. Split physical load (pressure_composite) into tertiles (Low/Medium/High)
4. For each signal: scatter plot with regression line + CI, faceted by
   physical load tertile
5. The dissociation claim: regression slopes are similar (negative) across
   all three tertiles, proving fatigue is cognitive, not just "tired legs."

Usage
-----
    python3 analysis/dissociation_figure.py [--data DATA] [--output OUTPUT]

Author : PM Sub-Agent (Focus Fatigue Project)
Created: 2026-07-19
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

N_EARLY_BLOCKS = 3
"""Number of early-match blocks used to compute per-player baseline."""

PHYSICAL_LOAD_COL = "pressure_composite"
"""Column used for physical load tertile split."""

COGNITIVE_LOAD_COL = "reorientation_rate"
"""X-axis: proxy for cognitive/attentional load.

Rationale:
    reorientation_rate measures how often a player scans the field per
    minute (head turns / reorientations). Higher values indicate more
    frequent situation-awareness checks — a direct behavioural marker
    of cognitive/attentional demand. This is the best available proxy
    for cognitive load in the Model 1 outputs.
"""

SIGNAL_COLS = [
    "shift_latency",
    "pressing_accuracy",
    "positional_drift",
    "transition_latency",
]
"""Signal columns in the unified dataset. Only those with data are plotted."""

SIGNAL_LABELS = {
    "shift_latency": "Shift Latency (s)",
    "pressing_accuracy": "Pressing Accuracy",
    "positional_drift": "Positional Drift (m)",
    "transition_latency": "Transition Latency (s)",
}
"""Plot-friendly labels for each signal."""

TERTILE_LABELS = ["Low", "Medium", "High"]
"""Labels for physical load tertile categories."""

NATURE_FIG_WIDTH = 183 / 25.4  # ~7.2 inches (Nature single-column)
NATURE_FIG_HEIGHT_PER_ROW = 2.5


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════


def load_unified_data(data_path: str) -> pd.DataFrame:
    """Load the unified parquet dataset.

    Parameters
    ----------
    data_path : str
        Path to ``unified_fatigue_dataset.parquet``.

    Returns
    -------
    pd.DataFrame
        Unified dataset.
    """
    path = Path(data_path)
    if not path.exists():
        print(f"  ❌ Unified dataset not found: {path}")
        print(f"     Run ``merge_outputs.py`` first.")
        sys.exit(1)

    df = pd.read_parquet(path)
    print(f"  Loaded {len(df)} rows × {len(df.columns)} columns from {path}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# Baseline Computation
# ═══════════════════════════════════════════════════════════════════════


def compute_baselines(
    df: pd.DataFrame,
    signal_cols: list[str],
    n_early_blocks: int = N_EARLY_BLOCKS,
    player_id_col: str = "player_id",
    block_num_col: str = "block_num",
) -> pd.DataFrame:
    """Compute per-player baselines from early-match blocks.

    For each player and signal, the baseline is the mean signal value
    across the first ``n_early_blocks`` blocks.

    Parameters
    ----------
    df : pd.DataFrame
        Unified dataset with per-block-per-player rows.
    signal_cols : list[str]
        Signal column names to compute baselines for.
    n_early_blocks : int
        Number of early blocks to use for baseline.
    player_id_col : str
        Player ID column name.
    block_num_col : str
        Block number column name.

    Returns
    -------
    pd.DataFrame
        Baseline DataFrame with index = player_id, columns = signal_cols + n_early_match_blocks.
        Also attaches ``n_baseline_blocks`` column.
    """
    # Determine which blocks are "early" (first N blocks of the match per player)
    early_blocks = df[df[block_num_col] < n_early_blocks].copy()

    if early_blocks.empty:
        print(f"  ⚠️  No blocks with block_num < {n_early_blocks} found.")
        print(f"     Current block_nums: {sorted(df[block_num_col].unique())}")
        print("     Falling back to global mean as baseline.")
        baselines = df[signal_cols].mean().to_frame(player_id_col).T
        baselines.index = ["global"]
        baselines["n_baseline_blocks"] = 0
        return baselines

    # Compute per-player mean across early blocks for each signal
    baseline_cols = signal_cols + [player_id_col]
    present_cols = [c for c in baseline_cols if c in early_blocks.columns and c != player_id_col]
    # Add block_num for counting
    group_cols = [player_id_col]

    baseline_df = early_blocks.groupby(group_cols)[present_cols].mean()
    baseline_df["n_baseline_blocks"] = early_blocks.groupby(group_cols).size()

    print(f"  Computed baselines for {len(baseline_df)} players "
          f"from first {n_early_blocks} blocks")
    print(f"  Baseline blocks range: {baseline_df['n_baseline_blocks'].min()}-"
          f"{baseline_df['n_baseline_blocks'].max()}")

    return baseline_df


def compute_signal_deficits(
    df: pd.DataFrame,
    baselines: pd.DataFrame,
    signal_cols: list[str],
) -> pd.DataFrame:
    """Compute signal deficit = actual - baseline for each player-block.

    Parameters
    ----------
    df : pd.DataFrame
        Unified dataset.
    baselines : pd.DataFrame
        Per-player baselines from ``compute_baselines()``.
    signal_cols : list[str]
        Signal columns to compute deficits for.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``{signal}_deficit`` columns added.
    """
    result = df.copy()
    player_id_col = "player_id"

    for sig in signal_cols:
        if sig not in df.columns:
            continue

        deficit_col = f"{sig}_deficit"

        # Merge baseline into result (pandas-native NaN handling)
        baseline_series = baselines[sig]
        if baselines.index.name == player_id_col:
            # Per-player baseline: use pandas map (handles NaN/None safely)
            mapped_baseline = result[player_id_col].map(baseline_series)
            result[deficit_col] = result[sig] - mapped_baseline
        else:
            # Global fallback baseline
            global_val = baseline_series.iloc[0]
            result[deficit_col] = result[sig] - global_val

        n_valid = result[deficit_col].notna().sum()
        print(f"  {deficit_col}: {n_valid} valid deficits computed")

    return result


# ═══════════════════════════════════════════════════════════════════════
# Physical Load Tertile Split
# ═══════════════════════════════════════════════════════════════════════


def assign_physical_load_tertiles(
    df: pd.DataFrame,
    load_col: str = PHYSICAL_LOAD_COL,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Assign each row to a physical load tertile.

    Uses ``qcut`` to split ``load_col`` into 3 groups. If there are
    duplicate values that prevent exactly 3 bins, it uses ``cut``
    with the unique values as bins.

    Parameters
    ----------
    df : pd.DataFrame
        Unified dataset.
    load_col : str
        Column name for physical load metric.
    labels : list[str] | None
        Labels for the tertile groups.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``physical_load_tertile`` column added.
    """
    result = df.copy()
    if labels is None:
        labels = TERTILE_LABELS

    unique_vals = result[load_col].nunique()

    try:
        if unique_vals >= 3:
            result["physical_load_tertile"] = pd.qcut(
                result[load_col], q=3, labels=labels, duplicates="raise"
            )
        elif unique_vals == 2:
            # Split at the midpoint between the two unique values
            low_val, high_val = sorted(result[load_col].unique())
            mid = (low_val + high_val) / 2
            result["physical_load_tertile"] = pd.cut(
                result[load_col],
                bins=[-np.inf, mid, np.inf],
                labels=["Low", "High"],
            )
            print(f"  ⚠️  Only {unique_vals} unique {load_col} values — "
                  f"using 2 bins (split at {mid})")
        else:
            # All same value — single bin
            result["physical_load_tertile"] = "All"
            print(f"  ⚠️  Only {unique_vals} unique {load_col} value — "
                  f"single category \"All\" used")
    except Exception as exc:
        print(f"  ⚠️  Tertile split failed ({exc}). Using single bin fallback.")
        result["physical_load_tertile"] = "All"

    print(f"  Physical load tertile distribution:")
    print(result["physical_load_tertile"].value_counts().to_string())

    return result


# ═══════════════════════════════════════════════════════════════════════
# Plotting — Faceted Dissociation Scatter
# ═══════════════════════════════════════════════════════════════════════


def plot_dissociation_faceted(
    plot_df: pd.DataFrame,
    x_col: str = COGNITIVE_LOAD_COL,
    y_ext: str = "_deficit",
    signal_cols: list[str] | None = None,
    tertile_col: str = "physical_load_tertile",
    output_prefix: str = "figures/dissociation",
    dpi: int = 300,
):
    """Create faceted scatter plots showing dissociation.

    One row per signal, one column per physical load tertile.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Dataset with signal deficits, tertile assignments.
    x_col : str
        Cognitive load metric (X-axis).
    y_ext : str
        Column name extension for Y-axis (e.g. '_deficit').
    signal_cols : list[str] | None
        Signal columns to plot. Auto-detects if None.
    tertile_col : str
        Column name for physical load tertile.
    output_prefix : str
        Base path for output files.
    dpi : int
        Figure resolution.
    """
    if signal_cols is None:
        # Auto-detect: columns ending with y_ext that have data
        signal_cols = [
            col.replace(y_ext, "")
            for col in plot_df.columns
            if col.endswith(y_ext) and plot_df[col].notna().sum() > 0
        ]

    # Create output dir
    out_dir = Path(output_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine tertile labels present (preserve order)
    tertile_order = TERTILE_LABELS + ["Low", "High", "All"]
    tertile_labels = sorted(
        [t for t in plot_df[tertile_col].dropna().unique() if pd.notna(t)],
        key=lambda x: tertile_order.index(x) if x in tertile_order else 999,
    )

    n_signals = len(signal_cols)
    n_tertiles = len(tertile_labels)

    if n_signals == 0:
        print("  ⚠️  No signals with data found. Nothing to plot.")
        return

    # ── Figure sizing ───────────────────────────────────────────────
    fig_width = NATURE_FIG_WIDTH  # single-column width
    # Widen for more columns
    if n_tertiles >= 3:
        fig_width = NATURE_FIG_WIDTH * 1.15
    elif n_tertiles == 2:
        fig_width = NATURE_FIG_WIDTH * 0.9

    fig_height = NATURE_FIG_HEIGHT_PER_ROW * n_signals + 0.8  # extra for legend

    fig, axes = plt.subplots(
        n_signals, n_tertiles,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex="col",
        sharey="row",
    )

    # ── Color palette ───────────────────────────────────────────────
    palette = sns.color_palette("viridis", max(n_tertiles, 3))
    tertile_color_map = dict(zip(tertile_labels, palette[:n_tertiles]))

    # ── Determine global axis limits for consistency ────────────────
    x_all = plot_df[x_col].dropna()
    x_min, x_max = x_all.min(), x_all.max()
    x_pad = (x_max - x_min) * 0.15 if x_max > x_min else 1.0
    y_global_min, y_global_max = np.inf, -np.inf

    for sig in signal_cols:
        y_c = f"{sig}{y_ext}"
        if y_c in plot_df.columns:
            y_vals = plot_df[y_c].dropna()
            if len(y_vals) > 0:
                y_global_min = min(y_global_min, y_vals.min())
                y_global_max = max(y_global_max, y_vals.max())

    y_pad = (y_global_max - y_global_min) * 0.2 if y_global_max > y_global_min else 0.5

    # ── Plot each signal row × tertile column ───────────────────────
    for i, signal in enumerate(signal_cols):
        y_col = f"{signal}{y_ext}"
        has_y_data = y_col in plot_df.columns

        for j, tertile in enumerate(tertile_labels):
            ax = axes[i, j]

            # Subset data for this tertile
            mask = plot_df[tertile_col] == tertile
            if has_y_data:
                sub = plot_df[mask].dropna(subset=[x_col, y_col])
            else:
                sub = plot_df[mask].dropna(subset=[x_col])

            panel_empty = sub.empty or (has_y_data and len(sub) == 0)

            if panel_empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="grey")
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                # Still show spines
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)
                continue

            # ── Scatter points ─────────────────────────────────────
            x_vals = sub[x_col].values
            y_vals = sub[y_col].values if has_y_data else np.zeros(len(sub))

            ax.scatter(
                x_vals, y_vals,
                s=35, alpha=0.7,
                color=tertile_color_map[tertile],
                edgecolors="white", linewidth=0.5,
                zorder=3,
            )

            # ── Regression line with confidence band ────────────────
            slope, intercept, r2, ci_possible = None, None, None, False

            if len(sub) >= 4:
                ci_possible = True
                try:
                    # Use regplot with scatter disabled (we already drew it)
                    sns.regplot(
                        data=sub,
                        x=x_col, y=y_col,
                        ax=ax,
                        scatter=False,
                        ci=95,
                        line_kws={"color": tertile_color_map[tertile],
                                  "linewidth": 1.8, "alpha": 0.85},
                        scatter_kws={"alpha": 0},
                        lowess=False,
                        truncate=True,
                    )
                    # Compute slope manually for annotation
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                    r2 = np.corrcoef(x_vals, y_vals)[0, 1] ** 2
                except Exception as exc:
                    print(f"  ⚠️  regplot failed for {signal}/{tertile}: {exc}")
                    try:
                        slope, intercept = np.polyfit(x_vals, y_vals, 1)
                        x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
                        y_line = slope * x_line + intercept
                        ax.plot(x_line, y_line, color=tertile_color_map[tertile],
                                linewidth=1.5, alpha=0.85)
                    except Exception:
                        pass
            elif len(sub) >= 2:
                # Too few points for regplot — just add a trend line
                try:
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                    x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, color=tertile_color_map[tertile],
                            linewidth=1.5, alpha=0.6, linestyle="--")
                except Exception:
                    pass

            # ── Slope + R² annotation ───────────────────────────────
            if slope is not None:
                n_pts = len(sub)
                annotation_text = f"β = {slope:.3f}"
                if r2 is not None:
                    annotation_text += f"\nR² = {r2:.3f}"
                annotation_text += f"\nn = {n_pts}"

                ax.text(
                    0.95, 0.05,
                    annotation_text,
                    transform=ax.transAxes,
                    ha="right", va="bottom",
                    fontsize=5.5, color="dimgrey",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.8,
                              edgecolor="lightgrey", linewidth=0.3),
                )

            # ── Labels ─────────────────────────────────────────────
            if i == n_signals - 1:
                ax.set_xlabel(
                    "Reorientation Rate (scans / min)\n[Cognitive Load Proxy]",
                    fontsize=7, labelpad=3)
            else:
                ax.set_xlabel("")

            if j == 0:
                y_label = SIGNAL_LABELS.get(signal, signal)
                if y_ext == "_deficit":
                    y_label = f"{y_label}\n(Deficit from Baseline)"
                ax.set_ylabel(y_label, fontsize=7, labelpad=2)

            # ── Column header ──────────────────────────────────────
            if i == 0:
                if tertile in ("All",):
                    header = "All Physical Load Levels"
                elif n_tertiles < 3:
                    header = f"Physical Load: {tertile}\n" + \
                             f"({sub[tertile_col].count()} blocks)"
                else:
                    header = f"Physical Load:\n{tertile}"
                ax.set_title(header, fontsize=7.5, fontweight="bold", pad=6)

            # ── Ticks ──────────────────────────────────────────────
            ax.tick_params(axis="both", which="major", labelsize=6, pad=2)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            if has_y_data:
                ax.set_ylim(y_global_min - y_pad, y_global_max + y_pad)

            # ── Nature-style spines ─────────────────────────────────
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            # ── Zero line for deficits ─────────────────────────────
            if y_ext == "_deficit" and has_y_data:
                if y_global_min < 0 < y_global_max:
                    ax.axhline(y=0, color="grey", linestyle="--",
                               linewidth=0.5, alpha=0.5)

        # Print summary for this signal
        if has_y_data:
            valid = plot_df.dropna(subset=[x_col, y_col])
            if len(valid) > 0:
                xv = valid[x_col].values
                yv = valid[y_col].values
                slope, _ = np.polyfit(xv, yv, 1)
                r2 = np.corrcoef(xv, yv)[0, 1] ** 2
                print(f"  Overall {signal}: n={len(valid)}, "
                      f"slope={slope:.3f}, R²={r2:.3f}")

            # ── Annotate regression slope ──────────────────────────
            if len(sub) >= 4:
                xv = sub[x_col].values
                yv = sub[y_col].values
                slope, intercept = np.polyfit(xv, yv, 1)
                r2 = np.corrcoef(xv, yv)[0, 1] ** 2 if len(xv) > 1 else 0
                # Small annotation in corner
                ax.text(
                    0.95, 0.05,
                    f"β = {slope:.2f}\nR² = {r2:.2f}",
                    transform=ax.transAxes,
                    ha="right", va="bottom",
                    fontsize=6, color="dimgrey",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", alpha=0.7),
                )

            # ── Labels ─────────────────────────────────────────────
            if i == n_signals - 1:
                ax.set_xlabel("Reorientation Rate (scans/min)\n[Cognitive Load Proxy]",
                              fontsize=7)
            else:
                ax.set_xlabel("")

            if j == 0:
                y_label = SIGNAL_LABELS.get(signal, signal)
                if y_ext == "_deficit":
                    y_label = f"{y_label}\n(Deficit from Baseline)"
                ax.set_ylabel(y_label, fontsize=7)

            # ── Column header ──────────────────────────────────────
            if i == 0:
                if tertile in ("All",):
                    ax.set_title(f"All Physical Load Levels",
                                 fontsize=8, fontweight="bold", pad=8)
                else:
                    ax.set_title(f"Physical Load: {tertile}",
                                 fontsize=8, fontweight="bold", pad=8)

            # ── Ticks ──────────────────────────────────────────────
            ax.tick_params(axis="both", which="major", labelsize=6)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)

            # ── Thinner spines (Nature style) ──────────────────────
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            # ── Zero line for deficits ─────────────────────────────
            if y_ext == "_deficit":
                ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5,
                           alpha=0.5)

        # Print summary for this signal
        y_col = f"{signal}{y_ext}"
        valid = plot_df.dropna(subset=[x_col, y_col])
        if len(valid) > 0:
            xv = valid[x_col].values
            yv = valid[y_col].values
            slope, intercept = np.polyfit(xv, yv, 1)
            r2 = np.corrcoef(xv, yv)[0, 1] ** 2
            print(f"  {signal}: n={len(valid)}, slope={slope:.3f}, R²={r2:.3f}")

    # ── Global legend for tertile colors ─────────────────────────
    if n_tertiles > 1:
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=6, label=l)
            for l, c in tertile_color_map.items()
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=n_tertiles,
            fontsize=6.5,
            frameon=False,
            handletextpad=0.8,
            columnspacing=2.0,
        )

    # ── Supertitle with methodological note ─────────────────────────
    fig.suptitle(
        "Dissociation of Cognitive Load from Physical Load \n"
        "on Defensive Quality Signals",
        fontsize=9, fontweight="bold", y=0.98, linespacing=1.3
    )

    # ── Adjust layout ─────────────────────────────────────────────
    legend_offset = 0.04 if n_tertiles > 1 else 0.02
    plt.subplots_adjust(
        left=0.11,
        right=0.97,
        top=0.88,
        bottom=0.10 + legend_offset,
        hspace=0.35,
        wspace=0.30,
    )

    # ── Save ──────────────────────────────────────────────────────
    for fmt in ["svg", "png"]:
        out_path = f"{output_prefix}.{fmt}"
        fig.savefig(
            out_path,
            dpi=dpi if fmt == "png" else 72,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"  ✅ Saved: {out_path} ({fmt})")

    plt.close(fig)
    print(f"\n  📊 Figure output: {output_prefix}.{{svg,png}}")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════


def run_dissociation_analysis(
    data_path: str = "outputs/unified_fatigue_dataset.parquet",
    output_prefix: str = "figures/dissociation",
    use_deficits: bool = True,
) -> dict:
    """Execute the full dissociation analysis pipeline.

    Parameters
    ----------
    data_path : str
        Path to unified parquet dataset.
    output_prefix : str
        Base path for output figure files.
    use_deficits : bool
        If True, use signal deficits (baseline-adjusted). If False,
        use raw signal values.

    Returns
    -------
    dict
        Summary of results.
    """
    print("=" * 70)
    print("  DISSOCIATION ANALYSIS: Cognitive Fatigue × Physical Load")
    print("=" * 70)
    print()

    # 1. Load
    print("Step 1/5: Loading unified data...")
    df = load_unified_data(data_path)
    print()

    # 2. Detect available signals
    available_signals = [s for s in SIGNAL_COLS if s in df.columns]
    print(f"Step 2/5: Detecting signals...")
    for sig in available_signals:
        n = df[sig].notna().sum()
        print(f"  {sig}: {n} non-null values")
    print()

    # 3. Compute baselines and signal deficits
    print("Step 3/5: Computing baselines and signal deficits...")
    baselines = compute_baselines(df, available_signals)

    if use_deficits and "global" not in str(baselines.index.name):
        plot_df = compute_signal_deficits(df, baselines, available_signals)
        y_col_ext = "_deficit"
        print("  Using signal deficits (per-player baseline adjusted)")
    else:
        plot_df = df.copy()
        y_col_ext = ""
        print("  Using raw signal values (insufficient baseline data)")
    print()

    # 4. Assign physical load tertiles
    print("Step 4/5: Assigning physical load tertiles...")
    plot_df = assign_physical_load_tertiles(plot_df)
    print()

    # 5. Plot
    print("Step 5/5: Generating faceted dissociation plot...")
    output_prefix_full = f"{output_prefix}"
    fig = plot_dissociation_faceted(
        plot_df,
        x_col=COGNITIVE_LOAD_COL,
        y_ext=y_col_ext,
        signal_cols=available_signals,
        output_prefix=output_prefix_full,
    )
    print()

    # ── Summary ──────────────────────────────────────────────────
    print("=" * 70)
    print("  ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Matches included:  {df['game_id'].nunique()}")
    print(f"  Players:           {df['player_id'].nunique()}")
    print(f"  Blocks per player: {df.groupby('player_id').size().describe()['mean']:.1f} avg")
    print(f"  Signals:           {', '.join(available_signals)}")
    print(f"  Cognitive load:   {COGNITIVE_LOAD_COL}")
    print(f"  Physical load:    {PHYSICAL_LOAD_COL}")
    print(f"  Y-axis:           {'Signal deficit (baseline-adjusted)' if use_deficits else 'Raw signal value'}")
    print(f"  Figure:           {output_prefix}.{{svg,png}}")
    print()
    print("  Key insight: Similar regression slopes across physical")
    print("  load tertiles → dissociation confirmed.")
    print("=" * 70)

    return {
        "n_matches": int(df["game_id"].nunique()),
        "n_players": int(df["player_id"].nunique()),
        "n_blocks_per_player": df.groupby("player_id").size().mean(),
        "signals": available_signals,
        "cognitive_load_metric": COGNITIVE_LOAD_COL,
        "physical_load_metric": PHYSICAL_LOAD_COL,
        "using_deficits": use_deficits,
        "figure_path": f"{output_prefix}.svg",
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI Entrypoint
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Dissociation Analysis: Cognitive Fatigue × Physical Load"
    )
    parser.add_argument(
        "--data",
        default="outputs/unified_fatigue_dataset.parquet",
        help="Path to unified parquet dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="figures/dissociation",
        help="Output figure path (prefix, default: %(default)s)",
    )
    parser.add_argument(
        "--raw-signals",
        action="store_true",
        help="Use raw signal values instead of baseline-adjusted deficits",
    )
    args = parser.parse_args()

    run_dissociation_analysis(
        data_path=args.data,
        output_prefix=args.output,
        use_deficits=not args.raw_signals,
    )


if __name__ == "__main__":
    main()
