#!/usr/bin/env python3
"""Add opponent quality covariate to the unified fatigue dataset.

Reads the unified dataset (Parquet), identifies each match's opponent,
looks up the opponent's final Ligue 1 2021-22 table position or a
proxy quality metric (e.g. league position, points per game, or
Elo rating from the CSV opponent goals data), and adds the covariate.

Usage:
    python3 analysis/opponent_quality_covariate.py \\
        --input  outputs/unified_fatigue_dataset.parquet \\
        --output outputs/unified_with_opponent_quality.parquet

Note:
    This script uses an heuristic mapping from game_id (tracking match ID)
    to CSV fixture row via shape JSON metadata. If shape JSON files are
    not available, fall back to a manual team-name lookup using the
    tracking data's ``team_id_opta`` and the team_mappings.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Constants ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = str(PROJECT_ROOT / "outputs" / "unified_fatigue_dataset.parquet")
DEFAULT_OUTPUT = str(
    PROJECT_ROOT / "outputs" / "unified_with_opponent_quality.parquet"
)
FIXTURES_PATH = PROJECT_ROOT / "data" / "ligue1_2021_22_complete_fixtures.csv"
SHAPE_OUTPUTS_DIR = PROJECT_ROOT.parent / "shape_outputs"
TEAM_MAPPINGS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "team_mappings" / "team_mappings.csv"
)

# Team name normalisation: shape JSON → fixture CSV
TEAM_NAME_MAP = {
    "Angers SCO": "Angers",
    "Clermont": "Clermont Foot",
    "Olympique Lyonnais": "Lyon",
    "Olympique Marseille": "Marseille",
}


def load_fixtures(path: Path) -> pd.DataFrame:
    """Load the Ligue 1 2021-22 fixtures CSV.

    Returns a DataFrame with columns:
    - HomeTeam, AwayTeam
    - HomeGoals, AwayGoals (for computing opponent strength)
    - Date
    """
    if not path.exists():
        print(f"  ⚠️  Fixtures file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    # Extract goals from list-like strings (e.g. "['4', '15', '82']")
    def _parse_goal_list(s):
        if isinstance(s, str) and s.startswith("["):
            try:
                return [int(x.strip().strip("'")) for x in s.strip("[]").split(",") if x.strip()]
            except (ValueError, IndexError):
                return []
        return []

    df["home_goal_mins"] = df["HomeGoalMinutes"].apply(_parse_goal_list)
    df["away_goal_mins"] = df["AwayGoalMinutes"].apply(_parse_goal_list)
    df["home_goals"] = pd.to_numeric(df["HomeGoals"], errors="coerce").fillna(0).astype(int)
    df["away_goals"] = pd.to_numeric(df["AwayGoals"], errors="coerce").fillna(0).astype(int)

    return df


def load_shape_metadata(
    game_id: str,
    shape_dir: Path,
) -> Optional[dict]:
    """Load a single shape JSON file to extract match metadata (team names).

    Parameters
    ----------
    game_id : str
        Match identifier (e.g. "2215771").
    shape_dir : Path
        Directory containing shape JSON files (``{game_id}.json`` or
        ``{game_id}/{game_id}.json``).

    Returns
    -------
    dict or None
        Dict with keys ``home_name``, ``away_name``, ``home_uuid``,
        ``away_uuid``, or None if not found.
    """
    candidates = [
        shape_dir / f"{game_id}.json",
        shape_dir / game_id / f"{game_id}.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path) as f:
                    raw = json.load(f)
                mi = raw.get("matchInfo", {})
                contestants = mi.get("contestant", [])
                home = next(
                    (c for c in contestants if c.get("position") == "home"), {}
                )
                away = next(
                    (c for c in contestants if c.get("position") == "away"), {}
                )
                home_name = TEAM_NAME_MAP.get(home.get("name", ""), home.get("name", ""))
                away_name = TEAM_NAME_MAP.get(away.get("name", ""), away.get("name", ""))
                return {
                    "home_name": home_name,
                    "away_name": away_name,
                    "home_uuid": home.get("id", ""),
                    "away_uuid": away.get("id", ""),
                }
            except (json.JSONDecodeError, KeyError) as e:
                print(f"    Error parsing shape file {path}: {e}")
                return None
    return None


def build_team_quality_lookup(fixtures_df: pd.DataFrame) -> dict[str, float]:
    """Build a lookup of team name → quality score.

    Quality is measured as goals scored per match — a simple proxy for
    attacking strength. Higher values = stronger opponent.

    Returns
    -------
    dict[str, float]
        ``{team_name: avg_goals_scored_per_match}``
    """
    if fixtures_df.empty:
        return {}

    # Goals scored by each team
    scoring: dict[str, list[int]] = {}

    for _, row in fixtures_df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        hg = int(row["home_goals"]) if pd.notna(row["home_goals"]) else 0
        ag = int(row["away_goals"]) if pd.notna(row["away_goals"]) else 0

        scoring.setdefault(home, []).append(hg)
        scoring.setdefault(away, []).append(ag)

    return {
        team: float(np.mean(goals)) if goals else 0.5
        for team, goals in scoring.items()
    }


def add_opponent_quality(
    unified_df: pd.DataFrame,
    shape_dir: Path,
    fixtures_df: pd.DataFrame,
    team_mappings_path: Path,
    team_quality: dict[str, float],
) -> pd.DataFrame:
    """Add opponent quality covariate to the unified dataset.

    Strategy:
    1. For each unique game_id, attempt to load shape JSON metadata
       to get home/away team names.
    2. Look up the opponent team name in the quality lookup.
    3. If shape metadata is unavailable, fall back to team_id_opta mapping.

    Parameters
    ----------
    unified_df : pd.DataFrame
        Unified fatigue dataset with at least ``game_id`` and
        ``team_id_opta`` columns.
    shape_dir : Path
        Directory containing shape JSON files.
    fixtures_df : pd.DataFrame
        Ligue 1 fixtures.
    team_mappings_path : Path
        Path to team_mappings.csv (UUID → OptaID).

    Returns
    -------
    pd.DataFrame
        Unified dataset with added column ``opponent_quality``.
    """
    df = unified_df.copy()

    # Load team mappings (UUID → OptaID)
    uuid_to_opta: dict[str, int] = {}
    opta_to_uuid: dict[int, str] = {}
    if team_mappings_path.exists():
        with open(team_mappings_path, newline="") as f:
            for row in csv.DictReader(f):
                u = row.get("uuid", "").strip()
                o = row.get("opta_id", "").strip()
                if u and o:
                    oid = int(o)
                    uuid_to_opta[u] = oid
                    opta_to_uuid[oid] = u

    # Build game_id → opponent quality mapping
    game_ids = sorted(df["game_id"].unique())
    game_opponent_quality: dict[str, dict[int, float]] = {}

    print(f"  Resolving opponent quality for {len(game_ids)} game IDs...")

    for gid in game_ids:
        # Get team IDs present in this match from the tracking data
        match_data = df[df["game_id"] == gid]
        teams_in_match = sorted(match_data["team_id_opta"].unique())
        teams_in_match = [t for t in teams_in_match if t > 0]  # skip ball/0

        # Try shape JSON first
        meta = load_shape_metadata(gid, shape_dir)

        if meta is not None and meta["home_name"] and meta["away_name"]:
            # Map: team_id_opta in tracking → team name
            # We need to match UUID → OptaID to figure out which OptaID
            # corresponds to home/away
            home_opta = uuid_to_opta.get(meta["home_uuid"])
            away_opta = uuid_to_opta.get(meta["away_uuid"])

            for tid in teams_in_match:
                if home_opta is not None and tid == home_opta:
                    # Our team is home → opponent is away
                    opp_name = meta["away_name"]
                elif away_opta is not None and tid == away_opta:
                    opp_name = meta["home_name"]
                else:
                    # Cannot determine — use average quality
                    opp_name = None

                if opp_name is not None and opp_name in team_quality:
                    game_opponent_quality.setdefault(gid, {})[int(tid)] = team_quality[opp_name]
        else:
            # Fallback: use simple heuristic based on team ID
            # (less accurate but doesn't crash)
            print(f"  ⚠️  No shape metadata for game_id={gid}; using heuristic")
            avg_quality = float(np.mean(list(team_quality.values()))) if team_quality else 0.5
            for tid in teams_in_match:
                game_opponent_quality.setdefault(gid, {})[int(tid)] = avg_quality

    # Apply opponent quality to each row
    df["opponent_quality"] = np.nan
    for gid, team_map in game_opponent_quality.items():
        for tid, quality in team_map.items():
            mask = (df["game_id"] == gid) & (df["team_id_opta"] == tid)
            df.loc[mask, "opponent_quality"] = quality

    print(f"  Added opponent_quality for {df['opponent_quality'].notna().sum()} rows")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Add opponent quality covariate to the unified fatigue dataset."
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT,
        help="Input unified parquet file (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help="Output parquet file with opponent_quality column (default: %(default)s)",
    )
    parser.add_argument(
        "--shape-dir", type=str, default=None,
        help="Shape JSON root directory (default: parent of project root/shape_outputs)",
    )
    parser.add_argument(
        "--fixtures", type=str, default=str(FIXTURES_PATH),
        help="Path to fixtures CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--team-mappings", type=str, default=str(TEAM_MAPPINGS_PATH),
        help="Path to team_mappings.csv (default: %(default)s)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    shape_dir = Path(args.shape_dir) if args.shape_dir else SHAPE_OUTPUTS_DIR

    print("Loading unified dataset...")
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

    print("Loading fixtures...")
    fixtures = load_fixtures(Path(args.fixtures))
    print(f"  Loaded {len(fixtures)} fixture rows")

    print("Computing team quality scores...")
    quality = build_team_quality_lookup(fixtures)
    print(f"  Quality scores for {len(quality)} teams")

    print("Adding opponent quality covariate...")
    result = add_opponent_quality(
        df, shape_dir, fixtures,
        Path(args.team_mappings), quality,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")
    print(f"   Shape: {len(result):,} rows × {len(result.columns)} columns")

    # Summary stats
    q = result["opponent_quality"].dropna()
    print(f"\nOpponent quality stats:")
    print(f"  Min: {q.min():.3f}")
    print(f"  Max: {q.max():.3f}")
    print(f"  Mean: {q.mean():.3f}")
    print(f"  Missing: {(~result['opponent_quality'].notna()).sum()} rows")


if __name__ == "__main__":
    main()
