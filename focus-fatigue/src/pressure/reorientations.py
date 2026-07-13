"""Reorientation Frequency — Load Indicator 3.

Detects sharp heading changes (>45° in <1s) using player velocity vectors.

Per-frame logic:
    1. Compute velocity heading: arctan2(vy, vx)
    2. For each frame, compute angular difference from frame n-25 frames back
       (1 second at 25fps).
    3. If angular difference > threshold AND player speed > min_speed → reorientation event.
    4. Count events per defender per block.

Uses smoothed velocities if available, otherwise raw speed_x/speed_y.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .config import PressureConfig, DEFAULT_CONFIG


def compute_velocity_heading(
    vx: np.ndarray,
    vy: np.ndarray,
) -> np.ndarray:
    """Compute heading angle (radians) from velocity components.

    Returns angles in [-π, π].
    """
    return np.arctan2(vy, vx)


def angular_difference(
    angle1: np.ndarray,
    angle2: np.ndarray,
) -> np.ndarray:
    """Compute signed angular difference, normalised to [-π, π].

    Positive = counterclockwise from angle1 to angle2.
    """
    diff = angle2 - angle1
    return (diff + np.pi) % (2 * np.pi) - np.pi


def detect_reorientations(
    df: pd.DataFrame,
    angle_threshold_deg: float = 45.0,
    window_s: float = 1.0,
    min_speed: float = 0.5,
    config: Optional[PressureConfig] = None,
    player_col: str = "player_id",
    frame_col: str = "frame_count",
    time_col: str = "time",
    speed_cols: tuple[str, str] = ("vx_smooth", "vy_smooth"),
    speed_mag_col: str = "v_mag",
    fallback_speed_cols: tuple[str, str] = ("speed_x", "speed_y"),
) -> pd.DataFrame:
    """Detect reorientation events per player.

    A reorientation is defined as a heading change exceeding the
    angle threshold within the specified time window, while the
    player is moving above minimum speed.

    Uses smoothed velocities (vx_smooth, vy_smooth) if available
    in the DataFrame, otherwise falls back to raw speed_x/speed_y.

    Parameters
    ----------
    df : pd.DataFrame
        Tracking data for a single match. May contain smoothed
        velocity columns from smoothing.smooth_trajectory.
    angle_threshold_deg : float, default 45.0
        Minimum absolute heading change (degrees) to count.
    window_s : float, default 1.0
        Time window in seconds for detecting changes.
    min_speed : float, default 0.5
        Minimum speed (m/s) — below this, player is "standing" and
        heading changes are not meaningful.
    config : PressureConfig, optional
        Overrides parameters if provided.
    player_col : str, default 'player_id'
    frame_col : str, default 'frame_count'
    time_col : str, default 'time'
    speed_cols : tuple, default ('vx_smooth', 'vy_smooth')
        Preferred velocity columns.
    speed_mag_col : str, default 'v_mag'
        Column for speed magnitude.
    fallback_speed_cols : tuple, default ('speed_x', 'speed_y')
        Fallback velocity columns if smoothed ones don't exist.

    Returns
    -------
    pd.DataFrame
        Original rows with added columns:
        - 'heading': velocity heading in radians
        - 'is_reorientation': bool flag for reorientation events
    """
    if config is not None:
        angle_threshold_deg = config.reorientation_angle_threshold
        window_s = config.reorientation_window_s
        min_speed = config.reorientation_min_speed

    angle_threshold_rad = np.radians(angle_threshold_deg)
    window_frames = int(round(window_s / 0.04))  # 25 frames @ 25fps

    result = df.copy()
    result["heading"] = np.nan
    result["is_reorientation"] = False

    # Determine velocity columns to use
    vx_col, vy_col = speed_cols
    if vx_col not in result.columns or vy_col not in result.columns:
        vx_col, vy_col = fallback_speed_cols

    # Determine speed magnitude column
    if speed_mag_col not in result.columns:
        # Compute from velocity components
        vx = result[vx_col].values
        vy = result[vy_col].values
        result["v_mag"] = np.sqrt(vx**2 + vy**2)
        speed_mag_col = "v_mag"

    for pid in result[player_col].unique():
        mask = result[player_col] == pid
        player_df = result.loc[mask].sort_values(frame_col)

        n = len(player_df)
        if n < window_frames + 1:
            continue

        # Get the velocity data
        vx = player_df[vx_col].values.astype(np.float64)
        vy = player_df[vy_col].values.astype(np.float64)
        v_mag = player_df[speed_mag_col].values.astype(np.float64)

        # Compute heading
        heading = compute_velocity_heading(vx, vy)

        # Store heading
        result.loc[player_df.index, "heading"] = heading

        # Detect reorientations: compare heading[t] vs heading[t - window]
        heading_prev = np.full(n, np.nan)
        heading_prev[window_frames:] = heading[:-window_frames]

        delta_heading = angular_difference(heading_prev, heading)
        abs_delta = np.abs(delta_heading)

        # Reorientation = big heading change + moving
        is_reo = (abs_delta >= angle_threshold_rad) & (v_mag >= min_speed)

        # Also require that the player was moving at the earlier point too
        v_mag_prev = np.full(n, np.nan)
        v_mag_prev[window_frames:] = v_mag[:-window_frames]
        is_reo = is_reo & (v_mag_prev >= min_speed)

        result.loc[player_df.index[is_reo], "is_reorientation"] = True

    return result


def aggregate_reorientations_to_blocks(
    blocks: list[pd.DataFrame],
    config: Optional[PressureConfig] = None,
) -> pd.DataFrame:
    """Count reorientation events per defender per block.

    Parameters
    ----------
    blocks : list of pd.DataFrame
        Tracking data split into blocks. Must have 'is_reorientation'.

    Returns
    -------
    pd.DataFrame
        One row per player per block with:
        block_id, player_id, team_id_opta, n_frames_reo,
        reorientation_count
    """
    records = []
    for blk in blocks:
        bid = blk["block_id"].iloc[0]
        # Exclude ball (player_id == -1)
        outfield = blk[blk["player_id"] != -1]
        if len(outfield) == 0:
            continue

        for (pid, tid), grp in outfield.groupby(["player_id", "team_id_opta"]):
            total_frames = len(grp)
            reo_count = int(grp["is_reorientation"].sum())
            records.append({
                "block_id": bid,
                "player_id": int(pid),
                "team_id_opta": int(tid),
                "n_frames_reo": total_frames,
                "reorientation_count": reo_count,
            })

    return pd.DataFrame(records)
