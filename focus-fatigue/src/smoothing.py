"""Trajectory smoothing using Savitzky-Golay filter.

The raw tracking data at 25fps is generally clean, but smoothing
helps reduce noise in derived velocity and acceleration metrics.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def smooth_trajectory(
    df: pd.DataFrame,
    window: int = 7,
    polyorder: int = 2,
    pos_cols: tuple[str, str] = ("x", "y"),
    player_col: str = "player_id",
    frame_col: str = "frame_count",
    inplace: bool = False,
) -> pd.DataFrame:
    """Apply Savitzky-Golay smoothing to player trajectories.

    Parameters
    ----------
    df : pd.DataFrame
        Tracking data for a single match, per-player per-frame.
    window : int, default 7
        Window length for Savitzky-Golay (must be odd).
    polyorder : int, default 2
        Polynomial order for Savitzky-Golay.
    pos_cols : tuple, default ('x', 'y')
        Column names for position coordinates.
    player_col : str, default 'player_id'
        Column identifying individual players.
    frame_col : str, default 'frame_count'
        Column for frame ordering.
    inplace : bool, default False
        If True, modify df in place by adding smoothed columns.
        If False, return a new DataFrame.

    Returns
    -------
    pd.DataFrame
        If inplace=False, new DataFrame with added columns:
        x_smooth, y_smooth, vx_smooth, vy_smooth
        If inplace=True, same but modified in place.
    """
    col_x, col_y = pos_cols

    if not inplace:
        df = df.copy()

    df["x_smooth"] = np.nan
    df["y_smooth"] = np.nan
    df["vx_smooth"] = np.nan
    df["vy_smooth"] = np.nan

    for player_id in df[player_col].unique():
        mask = df[player_col] == player_id
        player_df = df.loc[mask].sort_values(frame_col)

        n = len(player_df)
        if n < window:
            # Too few points — use raw values
            df.loc[mask, "x_smooth"] = player_df[col_x].values
            df.loc[mask, "y_smooth"] = player_df[col_y].values
            continue

        # Smooth x and y
        x_raw = player_df[col_x].values
        y_raw = player_df[col_y].values

        # Handle NaN values — replace with forward fill then linear
        x_clean = pd.Series(x_raw).interpolate(method="linear").ffill().bfill().values
        y_clean = pd.Series(y_raw).interpolate(method="linear").ffill().bfill().values

        x_smooth = savgol_filter(x_clean, window, polyorder)
        y_smooth = savgol_filter(y_clean, window, polyorder)

        # Compute smoothed velocities (derivative of smoothed positions)
        # Savitzky-Golay can also compute derivatives, but we use finite diff
        # on smoothed positions for clarity.
        vx = np.gradient(x_smooth)
        vy = np.gradient(y_smooth)

        # Scale to m/s (1 frame = 0.04s)
        dt = 0.04
        vx /= dt
        vy /= dt

        df.loc[mask, "x_smooth"] = x_smooth
        df.loc[mask, "y_smooth"] = y_smooth
        df.loc[mask, "vx_smooth"] = vx
        df.loc[mask, "vy_smooth"] = vy

    if not inplace:
        return df
    return None


def compute_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features from velocity to a tracking DataFrame.

    Expects columns: speed, speed_x, speed_y (from raw data)
    or vx_smooth, vy_smooth (from smoothing).

    Adds: total_speed (magnitude), heading, turn_rate
    """
    df = df.copy()

    # Use smoothed velocities if available, else raw
    if "vx_smooth" in df.columns:
        vx = df["vx_smooth"].values
        vy = df["vy_smooth"].values
    else:
        vx = df["speed_x"].values
        vy = df["speed_y"].values

    df["v_mag"] = np.sqrt(vx**2 + vy**2)
    df["heading"] = np.arctan2(vy, vx)

    # Turn rate: angular velocity (rad/s)
    # Computed per-player
    for pid in df["player_id"].unique():
        mask = df["player_id"] == pid
        heading = df.loc[mask, "heading"].values
        # Angular velocity = difference in heading / dt
        d_heading = np.diff(heading, prepend=heading[0])
        # Normalize to [-π, π]
        d_heading = (d_heading + np.pi) % (2 * np.pi) - np.pi
        df.loc[mask, "turn_rate"] = d_heading / 0.04

    return df
