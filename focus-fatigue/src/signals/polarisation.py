"""Signal — Team Polarisation.

Measures the degree of alignment of player movement vectors within each team
during out-of-possession phases. Computed as the mean resultant vector length
(R) of unit velocity vectors: R = 1 when all players move in perfect unison,
R = 0 when movement is completely random.

The core hypothesis is that fatigued teams exhibit reduced collective
coordination — defenders move less coherently, resulting in lower R values
in later match phases compared to early phases.

Reference
---------
- Mardia, K. V., & Jupp, P. E. (2000). *Directional Statistics*. Wiley.
  The mean resultant length R is the standard measure of concentration on
  the circle.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import SignalBase
from .registry import register_signal

# ── Constants ───────────────────────────────────────────────────────────────

_SIGNAL_NAME = "team_polarisation"
_BALL_PLAYER_ID = -1
_PITCH_LENGTH = 116.0
_PITCH_WIDTH = 78.0


# ═══════════════════════════════════════════════════════════════════════════
# Computation Functions
# ═══════════════════════════════════════════════════════════════════════════


def compute_polarisation_frame(
    frame_df: pd.DataFrame,
    team_in_possession_col: str = "team_in_possession",
    team_id_col: str = "team_id_opta",
    velocity_col: str = "v_mag",
    vx_col: str = "vx",
    vy_col: str = "vy",
    min_velocity: float = 0.1,
) -> dict[int, float]:
    """Compute team polarisation (mean resultant length R) for a single frame.

    Parameters
    ----------
    frame_df : pd.DataFrame
        Data for one frame (multiple players). Must contain columns:
        ``player_id``, ``team_id_opta``, ``vx``, ``vy`` (or ``vx_smooth``,
        ``vy_smooth``), and ``team_in_possession``.
    team_in_possession_col : str
        Column indicating which team has the ball.
    team_id_col : str
        Column for team ID.
    velocity_col : str
        Column containing velocity magnitude (used for filtering stationary).
    vx_col, vy_col : str
        Velocity component columns.
    min_velocity : float
        Minimum velocity magnitude to include a player in polarisation
        computation. Values below this threshold indicate a stationary
        player whose direction is undefined.

    Returns
    -------
    dict[int, float]
        Mapping of ``team_id_opta`` → mean resultant length R in [0, 1].
        Teams with fewer than 2 eligible players return NaN.
    """
    ball_mask = frame_df["player_id"] == _BALL_PLAYER_ID
    if ball_mask.any():
        in_poss = frame_df.loc[ball_mask, team_in_possession_col].iloc[0]
    else:
        in_poss = np.nan

    # Filter to outfield players only and determine out-of-possession teams
    outfield = frame_df[~ball_mask].copy()
    if len(outfield) == 0:
        return {}

    teams = outfield[team_id_col].unique()
    result: dict[int, float] = {}

    for team_id in teams:
        team_players = outfield[outfield[team_id_col] == team_id]

        # Only consider out-of-possession teams
        if not np.isnan(in_poss) and int(team_id) == int(in_poss):
            # Team in possession — skip (polarisation is for OOP)
            result[int(team_id)] = np.nan
            continue

        # Get unit velocity vectors
        vx = team_players[vx_col].values.astype(np.float64)
        vy = team_players[vy_col].values.astype(np.float64)
        speed = np.sqrt(vx**2 + vy**2)

        # Filter stationary players
        moving = speed >= min_velocity
        if moving.sum() < 2:
            result[int(team_id)] = np.nan
            continue

        # Unit vectors
        ux = vx[moving] / speed[moving]
        uy = vy[moving] / speed[moving]

        # Mean resultant vector length R
        sum_x = ux.sum()
        sum_y = uy.sum()
        n = float(len(ux))
        r = np.sqrt(sum_x**2 + sum_y**2) / n

        # Clip to [0, 1] to handle floating-point edge cases
        r = float(np.clip(r, 0.0, 1.0))
        result[int(team_id)] = r

    return result


def compute_polarisation_block(
    match_df: pd.DataFrame,
    block: dict[str, Any],
    team_in_possession_col: str = "team_in_possession",
    team_id_col: str = "team_id_opta",
    vx_col: str = "vx",
    vy_col: str = "vy",
    min_velocity: float = 0.1,
) -> list[dict[str, Any]]:
    """Compute mean polarisation per team for a single block.

    Optimised (vectorised) implementation using pandas groupby operations
    instead of per-frame Python loops (~20-50x speedup).

    Parameters
    ----------
    match_df : pd.DataFrame
        Full match tracking data.
    block : dict
        Block definition with keys ``block_id``, ``phase``, ``start_frame``,
        ``end_frame``.

    Returns
    -------
    list[dict]
        Records ready to be assembled into the output DataFrame.
    """
    start = block["start_frame"]
    end = block["end_frame"]

    frame_mask = match_df["frame_count"].between(start, end, inclusive="left")
    block_df = match_df[frame_mask]

    if len(block_df) == 0:
        return []

    # ── Step 1: Get per-frame ball possession ────────────────────────
    ball_df = block_df[block_df["player_id"] == _BALL_PLAYER_ID]
    if len(ball_df) > 0:
        in_poss_per_frame = (
            ball_df.set_index("frame_count")[team_in_possession_col].to_dict()
        )
    else:
        in_poss_per_frame = {}

    # ── Step 2: Filter to outfield, out-of-possession teams ─────────
    outfield = block_df[block_df["player_id"] != _BALL_PLAYER_ID].copy()
    if len(outfield) == 0:
        return []

    outfield["_in_poss"] = outfield["frame_count"].map(in_poss_per_frame)
    oop_mask = outfield["_in_poss"].isna() | (
        outfield[team_id_col] != outfield["_in_poss"]
    )
    oop = outfield[oop_mask].copy()

    if len(oop) == 0:
        return []

    # ── Step 3: Compute unit velocity vectors, filter stationary ────
    vx = oop[vx_col].values.astype(np.float64)
    vy = oop[vy_col].values.astype(np.float64)
    speed = np.sqrt(vx**2 + vy**2)
    moving_mask = speed >= min_velocity

    oop_moving = oop[moving_mask].copy()
    if len(oop_moving) < 2:
        return []

    sm = speed[moving_mask]
    oop_moving["_ux"] = oop_moving[vx_col].values / sm
    oop_moving["_uy"] = oop_moving[vy_col].values / sm

    # ── Step 4: Per (frame, team) compute mean resultant length R ───
    grouped = oop_moving.groupby(["frame_count", team_id_col], sort=False)
    sum_vx_g = grouped["_ux"].sum()
    sum_vy_g = grouped["_uy"].sum()
    counts = grouped["_ux"].count()

    # R = sqrt(sum_x^2 + sum_y^2) / n
    n_vals = counts.values.astype(np.float64)
    r_values = np.sqrt(sum_vx_g.values**2 + sum_vy_g.values**2) / n_vals

    # ── Step 5: Filter groups with < 2 moving players ────────────────
    valid = counts >= 2
    if not valid.any():
        return []

    r_valid = r_values[valid.values]
    valid_idx = [idx for i, idx in enumerate(sum_vx_g.index) if valid.iloc[i]]

    # Build a Series with valid R values indexed by (frame, team)
    r_series = pd.Series(r_valid, index=pd.MultiIndex.from_tuples(
        valid_idx, names=["frame_count", team_id_col]
    ))

    # ── Step 6: Aggregate per team (mean R across frames) ────────────
    team_r_mean = r_series.groupby(level=team_id_col, sort=False).mean()

    # Count valid frames per team (frames where >=2 moving players)
    team_nframes = (
        pd.Series(
            [1] * len(r_series),
            index=r_series.index
        )
        .groupby(level=team_id_col, sort=False)
        .sum()
    )

    # ── Step 7: Build records ────────────────────────────────────────
    records: list[dict[str, Any]] = []
    for team_id in team_r_mean.index:
        records.append({
            "block_id": block["block_id"],
            "phase": block["phase"],
            "player_id": 0,  # Team-level signal
            "team_id_opta": int(team_id),
            "signal_name": _SIGNAL_NAME,
            "signal_value": round(float(team_r_mean.loc[team_id]), 6),
            "n_frames": int(team_nframes.loc[team_id]),
        })

    return records

    return records


# ═══════════════════════════════════════════════════════════════════════════
# Block Conversion Helper
# ═══════════════════════════════════════════════════════════════════════════

def _blocks_to_dicts(blocks: list[Any]) -> list[dict[str, Any]]:
    """Convert blocks (list[DataFrame] or list[dict]) to list[dict].

    The pipeline passes ``blocks`` as ``list[pd.DataFrame]`` with a
    ``block_id`` column from ``split_into_blocks``. This function
    normalises them to the dict format needed for efficient frame-range
    filtering.
    """
    result: list[dict[str, Any]] = []
    for blk in blocks:
        if isinstance(blk, dict):
            bid = str(blk["block_id"])
            ph = int(blk.get("phase", bid.split("_")[0]))
            sf = int(blk.get("start_frame", 0))
            ef = int(blk.get("end_frame", 0))
        else:
            # DataFrame
            bid = str(blk["block_id"].iloc[0])
            ph = int(bid.split("_")[0])
            fc_col = "frame_count" if "frame_count" in blk.columns else "frame"
            sf = int(blk[fc_col].min())
            ef = int(blk[fc_col].max())
        result.append({"block_id": bid, "phase": ph,
                       "start_frame": sf, "end_frame": ef})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Registered Signal Class
# ═══════════════════════════════════════════════════════════════════════════

@register_signal
class PolarisationSignal(SignalBase):
    """Team polarisation signal — mean resultant vector length per team per block.

    Measures collective movement coherence for out-of-possession teams.
    High R indicates coordinated movement; low R indicates fragmented movement.
    """

    signal_name = _SIGNAL_NAME

    def __init__(
        self,
        signal_config=None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(config=signal_config, logger=logger)

    def compute(
        self,
        match_df: pd.DataFrame,
        blocks: list[Any],
        *,
        game_id: str = "",
        team_in_possession_col: str = "team_in_possession",
        team_id_col: str = "team_id_opta",
        min_velocity: float = 0.1,
    ) -> pd.DataFrame:
        """Compute team polarisation per block.

        Parameters
        ----------
        match_df : pd.DataFrame
            Full tracking data for one match. Must contain columns:
            ``player_id``, ``team_id_opta``, ``frame_count``, ``vx``, ``vy``,
            ``team_in_possession``.
        blocks : list[DataFrame | dict]
            Block definitions from ``split_into_blocks``.
        game_id : str
            Match identifier.
        team_in_possession_col : str
            Column indicating which team has the ball.
        team_id_col : str
            Column for team ID.
        min_velocity : float
            Minimum velocity magnitude to include a player.

        Returns
        -------
        pd.DataFrame
            Standard signal output with one row per team per block.
        """
        if len(match_df) == 0 or len(blocks) == 0:
            empty = pd.DataFrame(columns=[
                "game_id", "block_id", "phase", "player_id", "team_id_opta",
                "signal_name", "signal_value", "n_frames",
            ])
            return empty

        # Determine velocity columns
        vx_col = "vx_smooth" if "vx_smooth" in match_df.columns else "vx"
        vy_col = "vy_smooth" if "vy_smooth" in match_df.columns else "vy"
        if vx_col not in match_df.columns or vy_col not in match_df.columns:
            self.logger.warning(
                "Velocity columns (%s, %s) not found; "
                "falling back to zero-velocity assumption",
                vx_col, vy_col,
            )
            match_df = match_df.copy()
            match_df[vx_col] = 0.0
            match_df[vy_col] = 0.0

        # Normalise blocks to dicts
        block_dicts = _blocks_to_dicts(blocks)

        # Compute polarisation per block
        all_records: list[dict[str, Any]] = []
        for bd in block_dicts:
            records = compute_polarisation_block(
                match_df, bd,
                team_in_possession_col=team_in_possession_col,
                team_id_col=team_id_col,
                vx_col=vx_col, vy_col=vy_col,
                min_velocity=min_velocity,
            )
            for rec in records:
                rec["game_id"] = game_id
            all_records.extend(records)

        if not all_records:
            empty = pd.DataFrame(columns=[
                "game_id", "block_id", "phase", "player_id", "team_id_opta",
                "signal_name", "signal_value", "n_frames",
            ])
            return empty

        out = pd.DataFrame(all_records)

        # Ensure standard output columns
        out = self.ensure_output_columns(out)

        # Cast types
        for c in ["player_id", "team_id_opta", "phase", "n_frames"]:
            out[c] = out[c].astype(int)
        for c in ["game_id", "block_id", "signal_name"]:
            out[c] = out[c].astype(str)
        out["signal_value"] = out["signal_value"].astype(float)

        return out.reset_index(drop=True)

    def validate(self, output_df: pd.DataFrame) -> bool:
        """Validate polarisation output.

        Checks:
        - Standard schema compliance
        - signal_value in [0, 1] (mean resultant length R always in [0, 1])
        - n_frames non-negative
        """
        super().validate(output_df)
        if len(output_df) == 0:
            return True

        sv = output_df["signal_value"]
        if sv.min() < 0.0 or sv.max() > 1.0:
            raise ValueError(
                f"signal_value must be in [0, 1], got "
                f"[{sv.min():.6f}, {sv.max():.6f}]"
            )

        # n_frames must be non-negative
        if output_df["n_frames"].min() < 0:
            raise ValueError(
                f"n_frames contains negative values "
                f"(min={output_df['n_frames'].min()})"
            )

        # player_id should be 0 (team-level signal)
        unique_pids = output_df["player_id"].unique()
        if len(unique_pids) > 1 or unique_pids[0] != 0:
            self.logger.warning(
                "Polarisation signal is team-level; expected player_id=0, "
                "got %s", unique_pids
            )

        return True
