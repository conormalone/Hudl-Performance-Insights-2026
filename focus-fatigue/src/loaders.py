"""Data loaders — one file with all data-loading functions.

Purpose: Load tracking data and team mappings from disk.
Every other module imports from here, never from individual loader files.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# ── Constants ───────────────────────────────────────────────────────────────

PITCH_X_MIN = -57.0
PITCH_X_MAX = 59.0
PITCH_Y_MIN = -39.0
PITCH_Y_MAX = 39.0
FRAME_INTERVAL_S = 0.04
BALL_PLAYER_ID = -1

KEEP_COLS_RAW = [
    "current_phase", "timeelapsed", "team_id_opta", "player_id",
    "jersey_no", "pos_x", "pos_y", "speed", "frame_count",
    "speed_x", "speed_y", "acc", "dop", "team_in_possession",
]


# ── Tracking Data Loader ────────────────────────────────────────────────────


def _normalise_dtype(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["player_id", "team_id_opta"]:
        df[col] = df[col].fillna(-1).astype(np.int64)
    return df


def _canonicalise_coords(df: pd.DataFrame, normalise_dop: bool = True) -> pd.DataFrame:
    """Flip x-axis so all attacks go left→right (DOP=R).

    Uses per-player DOP (direction of play) to flip each player's coordinates
    individually. Within a frame, each team has a unanimous DOP value, so this
    is equivalent to the per-frame mode approach but avoids an expensive
    groupby-aggregate on ~143K frames (~60s speedup).
    """
    if not normalise_dop:
        return df
    flips = df["dop"].notna() & (df["dop"] == "L")
    df.loc[flips, "x"] *= -1
    df.loc[flips, "speed_x"] *= -1
    return df


def load_tracking_statsperform(
    filepath: Union[str, Path],
    match_id: Optional[str] = None,
    normalise_dop: bool = True,
    include_ball: bool = True,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a Stats Perform tracking.parquet file.

    Returns canonicalised tracking data with columns:
    game_id, phase, frame_count, time, player_id, team_id_opta,
    jersey_no, x, y, speed, speed_x, speed_y, acc, dop, team_in_possession
    """
    filepath = Path(filepath)
    if match_id is None:
        match_id = filepath.parent.name

    df = pd.read_parquet(filepath, columns=KEEP_COLS_RAW)  # column projection: reads only ~356MB instead of ~519MB
    if nrows is not None:
        df = df.head(nrows)
    df = _normalise_dtype(df)

    df = df.rename(columns={"current_phase": "phase", "timeelapsed": "time",
                             "pos_x": "x", "pos_y": "y"})
    df = _canonicalise_coords(df, normalise_dop=normalise_dop)
    df["game_id"] = match_id

    output_cols = [
        "game_id", "phase", "frame_count", "time", "player_id",
        "team_id_opta", "jersey_no", "x", "y", "speed", "speed_x",
        "speed_y", "acc", "dop", "team_in_possession",
    ]

    if not include_ball:
        # Fall back to load_tracking's GK heuristic
        from src.pressure.gk_utils import flag_goalkeepers
        df["is_goalkeeper"] = flag_goalkeepers(df)
        df = df[df["player_id"] != BALL_PLAYER_ID].copy()
        output_cols.append("is_goalkeeper")

    return df[output_cols].reset_index(drop=True)


def load_tracking_by_match(
    tracking_dir: Union[str, Path],
    match_ids: Optional[list[str]] = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Load multiple tracking files into {match_id: DataFrame}."""
    tracking_dir = Path(tracking_dir)
    if match_ids is None:
        match_ids = sorted(
            d.name for d in tracking_dir.iterdir()
            if d.is_dir() and (d / "tracking.parquet").exists()
        )
    return {
        mid: load_tracking_statsperform(tracking_dir / mid / "tracking.parquet",
                                         match_id=mid, **kwargs)
        for mid in match_ids
        if (tracking_dir / mid / "tracking.parquet").exists()
    }
