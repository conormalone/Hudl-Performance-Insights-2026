#!/usr/bin/env python3
"""Post-hoc analysis: re-run dissociation analysis with opponent quality covariate.

This script:
  1. Loads the unified dataset (outputs/unified_dataset.csv)
  2. Assigns opponent quality using a simple Elo approximation based on
     match results
  3. Re-runs dissociation analysis with opponent_quality as a covariate
  4. Saves results to analysis/opponent_quality_results.csv

Usage:
    python3 analysis/opponent_quality_covariate.py
    python3 analysis/opponent_quality_covariate.py --input outputs/unified_dataset.csv

Dependencies: pandas, numpy, scipy, statsmodels (optional for OLS)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR  # noqa: E402

# ── Elo parameters ─────────────────────────────────────────────────
ELO_K = 32          # K-factor for Elo updates
ELO_INITIAL = 1500  # Starting Elo for each team
ELO_HOME_ADV = 50   # Home advantage bonus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Opponent quality covariate analysis.",
    )
    parser.add_argument(
        "--input", "-i", type=str,
        default=str(OUTPUT_DIR / "unified_dataset.csv"),
        help="Path to unified dataset CSV",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default=str(PROJECT_ROOT / "analysis" / "opponent_quality_results.csv"),
        help="Output path for results CSV",
    )
    parser.add_argument(
        "--matches", "-m", type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "matches.csv"),
        help="Path to matches metadata CSV",
    )
    return parser


def load_matches(matches_path: Path) -> Optional[pd.DataFrame]:
    """Load match metadata.

    Expected columns: match_id, home_team, away_team, home_score, away_score
    """
    if not matches_path.exists():
        logger.warning("Match metadata not found at %s", matches_path)
        return None

    df = pd.read_csv(matches_path)
    logger.info("Loaded %d matches from %s", len(df), matches_path)
    return df


def compute_elo_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute Elo ratings for each team across all matches.

    Returns matches DataFrame with an additional 'opponent_elo' column
    representing the opponent's Elo rating at the time of the match.

    Simple Elo model:
      expected = 1 / (1 + 10^((rating_b - rating_a) / 400))
      new_rating = rating_a + K * (actual - expected)

    Where actual = 1 if win, 0.5 if draw, 0 if loss.
    """
    df = matches.copy()

    # Check required columns
    required = ["match_id", "home_team", "away_team"]
    for col in required:
        if col not in df.columns:
            logger.error("Missing required column '%s' in match metadata", col)
            return df

    # Sort by match_id to process chronologically
    df = df.sort_values("match_id").reset_index(drop=True)

    # Initialise team ratings
    ratings = {}
    opponent_elos = []
    home_elos = []

    for _, row in df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))

        if home not in ratings:
            ratings[home] = ELO_INITIAL + ELO_HOME_ADV
        if away not in ratings:
            ratings[away] = ELO_INITIAL

        # Store opponent Elos before updating
        opponent_elos.append(ratings[away])
        home_elos.append(ratings[home])

        # Determine result from scores (if available)
        if "home_score" in df.columns and "away_score" in df.columns:
            hs = row.get("home_score", 0)
            ask = row.get("away_score", 0)
            try:
                hs, ask = int(hs), int(ask)
            except (ValueError, TypeError):
                hs, ask = 0, 0

            if hs > ask:
                home_actual, away_actual = 1.0, 0.0
            elif hs < ask:
                home_actual, away_actual = 0.0, 1.0
            else:
                home_actual, away_actual = 0.5, 0.5
        else:
            # No scores: assume neutral
            home_actual, away_actual = 0.5, 0.5

        # Expected scores
        r_home = ratings[home]
        r_away = ratings[away]
        exp_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home) / 400.0))
        exp_away = 1.0 - exp_home

        # Update ratings
        ratings[home] = r_home + ELO_K * (home_actual - exp_home)
        ratings[away] = r_away + ELO_K * (away_actual - exp_away)

    df["home_elo"] = home_elos
    df["opponent_elo"] = opponent_elos
    logger.info("Elos computed: %d teams, range [%.0f, %.0f]",
                len(ratings), min(ratings.values()), max(ratings.values()))
    return df


def assign_opponent_quality(
    unified_df: pd.DataFrame,
    match_elo_df: Optional[pd.DataFrame],
    team_col: str = "team_id_opta",
    match_id_col: str = "game_id",
) -> pd.DataFrame:
    """Assign opponent quality to each row in the unified dataset.

    Uses Elo ratings from match metadata. Falls back to a simple
    team-ID proxy if Elo data is unavailable.

    The 'team' columns in unified_df indicate WHICH team's perspective
    the measurement is from. The opponent quality is the Elo of the
    OTHER team.
    """
    df = unified_df.copy()

    if match_elo_df is not None and not match_elo_df.empty:
        # Build a lookup: match_id -> (home_team, away_team, home_elo, opponent_elo)
        elo_map = {}
        for _, row in match_elo_df.iterrows():
            mid = row.get("match_id")
            elo_map[mid] = {
                "home_team": str(row.get("home_team", "")),
                "away_team": str(row.get("away_team", "")),
                "home_elo": row.get("home_elo", 1500),
                "opponent_elo": row.get("opponent_elo", 1500),
            }

        def lookup_opponent_elo(row):
            mid = row.get(match_id_col)
            team = str(row.get(team_col, ""))
            if mid in elo_map:
                m = elo_map[mid]
                if team == m["home_team"]:
                    return m["opponent_elo"]
                elif team == m["away_team"]:
                    return m["home_elo"]
            return 1500  # default

        df["opponent_quality"] = df.apply(lookup_opponent_elo, axis=1)
    else:
        # Fallback: use team_id as a proxy rank
        # Sort teams by their average signal value, use percentile rank
        logger.warning("No Elo data available — using team-ID proxy ranking")
        team_avgs = df.groupby(team_col)["signal_value"].mean()
        team_ranks = team_avgs.rank(pct=True)
        df["opponent_quality"] = df[team_col].map(team_ranks)
        df["opponent_quality"] = df["opponent_quality"].fillna(0.5)

    logger.info("Opponent quality assigned: mean=%.2f, std=%.2f",
                df["opponent_quality"].mean(), df["opponent_quality"].std())
    return df


def run_dissociation_with_covariate(
    df: pd.DataFrame,
    signal_name_col: str = "signal_name",
    signal_value_col: str = "signal_value",
    covariate_col: str = "opponent_quality",
    group_col: str = "block_id",
) -> pd.DataFrame:
    """Re-run dissociation analysis with opponent quality as covariate.

    For each signal, computes the partial correlation between block_id
    (as time proxy) and signal_value, controlling for opponent_quality.

    Returns a DataFrame with signal-level dissociation results.
    """
    from scipy.stats import pearsonr, spearmanr

    results = []
    signals = df[signal_name_col].unique()

    for signal in signals:
        sdf = df[df[signal_name_col] == signal].dropna(subset=[signal_value_col, covariate_col])
        if len(sdf) < 10:
            logger.warning("Signal '%s': insufficient data (%d rows)", signal, len(sdf))
            continue

        # Simple partial correlation via residualisation:
        # 1. Regress signal_value on opponent_quality
        # 2. Regress block_id on opponent_quality
        # 3. Correlate the residuals

        values = sdf[signal_value_col].values.astype(float)
        blocks = sdf[group_col].values.astype(float)
        covar = sdf[covariate_col].values.astype(float)

        # Add constant term
        X = np.column_stack([np.ones_like(covar), covar])

        # Residuals via OLS manually
        try:
            # signal ~ covariate
            beta_v = np.linalg.lstsq(X, values, rcond=None)[0]
            resid_v = values - X @ beta_v

            # block_id ~ covariate
            beta_b = np.linalg.lstsq(X, blocks, rcond=None)[0]
            resid_b = blocks - X @ beta_b

            # Correlation of residuals
            if np.std(resid_v) > 1e-10 and np.std(resid_b) > 1e-10:
                r_pearson, p_pearson = pearsonr(resid_v, resid_b)
                r_spearman, p_spearman = spearmanr(resid_v, resid_b)
            else:
                r_pearson, p_pearson = 0.0, 1.0
                r_spearman, p_spearman = 0.0, 1.0

        except np.linalg.LinAlgError:
            r_pearson, p_pearson = np.nan, np.nan
            r_spearman, p_spearman = np.nan, np.nan

        # Also compute the uncorrected (simple) correlations for comparison
        if np.std(values) > 1e-10 and np.std(blocks) > 1e-10:
            r_raw, p_raw = pearsonr(values, blocks)
        else:
            r_raw, p_raw = 0.0, 1.0

        results.append({
            "signal": signal,
            "n_observations": len(sdf),
            "r_pearson_covariate_corrected": r_pearson,
            "p_pearson_covariate_corrected": p_pearson,
            "r_spearman_covariate_corrected": r_spearman,
            "p_spearman_covariate_corrected": p_spearman,
            "r_pearson_uncorrected": r_raw,
            "p_pearson_uncorrected": p_raw,
            "covariate": covariate_col,
            "mean_opponent_quality": float(np.mean(covar)),
            "std_opponent_quality": float(np.std(covar)),
        })

    result_df = pd.DataFrame(results)
    logger.info("Dissociation analysis complete: %d signals", len(result_df))
    return result_df


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    matches_path = Path(args.matches)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        logger.error("Run the pipeline first to generate unified_dataset.csv")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("Opponent Quality Covariate Analysis")
    logger.info("=" * 50)

    # 1. Load unified dataset
    logger.info("Loading unified dataset from %s", input_path)
    unified = pd.read_csv(input_path)
    logger.info("Loaded %d rows x %d columns", len(unified), len(unified.columns))

    # 2. Load match metadata and compute Elo
    matches = load_matches(matches_path)
    if matches is not None:
        match_elo = compute_elo_ratings(matches)
    else:
        match_elo = None

    # 3. Assign opponent quality
    unified = assign_opponent_quality(unified, match_elo)

    # 4. Run dissociation analysis with covariate
    results = run_dissociation_with_covariate(unified)

    # 5. Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)
    logger.info("")
    logger.info("Summary:")
    logger.info(results.to_string(index=False))

    # Also save the enriched dataset
    enriched_path = output_path.parent / "unified_with_opponent_quality.csv"
    unified.to_csv(enriched_path, index=False)
    logger.info("Enriched dataset saved to %s", enriched_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
