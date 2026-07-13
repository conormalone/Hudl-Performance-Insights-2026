"""Data loader for Stats Perform optical tracking data.

Reads tracking.parquet files produced by Stats Perform's optical tracking system.
Provides coordinate canonicalisation (DOP-normalised) and dtype normalisation.

Columns (raw):
    current_phase, timeelapsed, team_id_opta, player_id, jersey_no,
    pos_x, pos_y, speed, frame_count, goalkeeper, team_id, acc,
    in_play, speed_x, speed_y, dop, team_in_possession

Columns (output):
    game_id, phase, frame, time, player_id, team_id_opta, jersey_no,
    x, y, speed, speed_x, speed_y, acc, dop_normalised, team_in_possession
"""

from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd

# --- Constants ---

# Stats Perform coordinate system: centred pitch
# x: [-57, 59], y: [-39, 39] = approx 116m x 78m
PITCH_X_MIN = -57.0
PITCH_X_MAX = 59.0
PITCH_Y_MIN = -39.0
PITCH_Y_MAX = 39.0

# Frame rate is exactly 25fps (0.04s intervals)
FRAME_INTERVAL_S = 0.04

# DOP (Direction of Play) markers
DOP_LEFT = "L"
DOP_RIGHT = "R"

# Ball marker — player_id == -1 in raw data
BALL_PLAYER_ID = -1

# Columns to keep from raw parquet
KEEP_COLS_RAW = [
    "current_phase", "timeelapsed", "team_id_opta", "player_id",
    "jersey_no", "pos_x", "pos_y", "speed", "frame_count",
    "speed_x", "speed_y", "acc", "dop", "team_in_possession",
]


def _normalise_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes across matches (some have float, some int).

    player_id: fill NaN with -1, cast to int
    team_id_opta: fill NaN with -1, cast to int
    """
    df = df.copy()
    for col in ["player_id", "team_id_opta"]:
        df[col] = df[col].fillna(-1).astype(np.int64)
    return df


def _resolve_goalkeeper(df: pd.DataFrame) -> pd.Series:
    """Flag goalkeepers based on jersey number (1) and position.

    The goalkeeper column in raw data is not reliably populated.
    We flag a player as GK if jersey_no == 1 AND their average
    position is close to their own goal line (extreme x values).
    This is a heuristic; shape.json role descriptions provide
    the ground truth (see load_shapes.py).
    """
    # Simple heuristic: players with jersey 1 whose mean pos_y
    # is near the centre and mean pos_x is at an extreme
    gk_candidates = (
        df[df["jersey_no"] == 1]
        .groupby(["player_id", "team_id_opta"])[["pos_x", "pos_y"]]
        .mean()
    )
    is_gk = {}
    for (pid, tid), row in gk_candidates.iterrows():
        # GK is typically within 10m of goal line and near centre
        near_goal = abs(row["pos_x"]) > 45  # near x=-57 or x=59 extremes
        near_center = abs(row["pos_y"]) < 10
        is_gk[(pid, tid)] = near_goal and near_center

    return df.apply(
        lambda r: is_gk.get((r["player_id"], r["team_id_opta"]), False), axis=1
    )


def _canonicalise_coords(
    df: pd.DataFrame,
    normalise_dop: bool = True,
    x_col: str = "x",
    speed_x_col: str = "speed_x",
) -> pd.DataFrame:
    """Canonicalise pitch coordinates.

    When normalise_dop=True, flip the x-axis so that all attacks
    go left → right (DOP=R). This makes teams comparable across
    matches regardless of which half they started in.

    For DOP=L: x *= -1, speed_x *= -1
    For DOP=R: leave as-is

    The frame-level dop is per-player/ball (usually consistent
    within a phase). We use the majority value per frame.
    """
    df = df.copy()

    if not normalise_dop:
        return df

    # Get DOP direction per frame from the ball
    ball_dop = (
        df[df["player_id"] == BALL_PLAYER_ID]
        .groupby("frame_count")["dop"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else DOP_RIGHT)
        .to_dict()
    )

    # Apply per-frame DOP normalisation
    flip_mask = df["frame_count"].map(
        lambda f: ball_dop.get(f, DOP_RIGHT) == DOP_LEFT
    )

    df.loc[flip_mask, x_col] *= -1
    df.loc[flip_mask, speed_x_col] *= -1

    return df


def load_tracking_statsperform(
    filepath: Union[str, Path],
    match_id: Optional[str] = None,
    normalise_dop: bool = True,
    include_ball: bool = True,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load a Stats Perform tracking.parquet file.

    Parameters
    ----------
    filepath : str or Path
        Path to the tracking.parquet file.
    match_id : str, optional
        Match identifier. If None, inferred from parent folder name.
    normalise_dop : bool, default True
        If True, flip coordinates so all attacks go left→right.
    include_ball : bool, default True
        If False, filter out ball rows (player_id == -1).
    columns : list of str, optional
        Subset of columns to return. If None, returns all output columns.

    Returns
    -------
    pd.DataFrame
        Canonicalised tracking data with columns:
        game_id, phase, frame, time, player_id, team_id_opta, jersey_no,
        x, y, speed, speed_x, speed_y, acc, team_in_possession
        (plus is_goalkeeper if not include_ball)
    """
    filepath = Path(filepath)
    if match_id is None:
        match_id = filepath.parent.name

    # Read raw parquet
    df = pd.read_parquet(filepath)
    df = _normalise_dtype(df)

    # Rename columns to output schema
    df = df.rename(
        columns={
            "current_phase": "phase",
            "timeelapsed": "time",
            "pos_x": "x",
            "pos_y": "y",
        }
    )

    # Canonicalise coordinates
    df = _canonicalise_coords(df, normalise_dop=normalise_dop)

    # Compute goalkeeper flag (heuristic)
    if not include_ball:
        df["is_goalkeeper"] = _resolve_goalkeeper(df)

    # Filter ball if requested
    if not include_ball:
        df = df[df["player_id"] != BALL_PLAYER_ID].copy()

    # Add game_id
    df["game_id"] = match_id

    # Select output columns
    output_cols = [
        "game_id",
        "phase",
        "frame_count",
        "time",
        "player_id",
        "team_id_opta",
        "jersey_no",
        "x",
        "y",
        "speed",
        "speed_x",
        "speed_y",
        "acc",
        "dop",
        "team_in_possession",
    ]
    if not include_ball:
        output_cols.append("is_goalkeeper")

    if columns:
        output_cols = [c for c in columns if c in output_cols]

    return df[output_cols].reset_index(drop=True)


def load_tracking_by_match(
    tracking_dir: Union[str, Path],
    match_ids: Optional[list[str]] = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Load multiple matches into a dict keyed by match ID.

    Parameters
    ----------
    tracking_dir : str or Path
        Directory containing subdirectories (one per match), each with
        tracking.parquet inside.
    match_ids : list of str, optional
        Subset of match IDs to load. If None, loads all.

    Returns
    -------
    dict[str, pd.DataFrame]
        {match_id: loaded_tracking_df}
    """
    tracking_dir = Path(tracking_dir)
    result = {}

    if match_ids is None:
        match_ids = sorted(
            d.name
            for d in tracking_dir.iterdir()
            if d.is_dir() and (d / "tracking.parquet").exists()
        )

    for mid in match_ids:
        fp = tracking_dir / mid / "tracking.parquet"
        if not fp.exists():
            continue
        result[mid] = load_tracking_statsperform(fp, match_id=mid, **kwargs)

    return result
