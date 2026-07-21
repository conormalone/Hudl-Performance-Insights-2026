"""Signal — Team Centroid Distance.

Measures how far each player is from their team's centroid (mean x, mean y)
during out-of-possession phases. The core hypothesis is that fatigued players
display degraded positional discipline — drifting further from their team's
collective shape — manifested as larger centroid distances compared to fresh
players earlier in the match.

This signal provides a per-player, per-block measure complementary to
team-level polarisation: while polarisation captures movement *direction*
coherence, centroid distance captures spatial *dispersion*.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import SignalBase
from .registry import register_signal

# ── Constants ───────────────────────────────────────────────────────────────

_SIGNAL_NAME = "team_centroid_distance"
_BALL_PLAYER_ID = -1


# ═══════════════════════════════════════════════════════════════════════════
# Computation Functions
# ═══════════════════════════════════════════════════════════════════════════


def compute_centroid_distance_frame(
    frame_df: pd.DataFrame,
    team_in_possession_col: str = "team_in_possession",
    team_id_col: str = "team_id_opta",
) -> dict[int, dict[int, float]]:
    """For a single frame, compute distance of each player to their team centroid.

    Operates only on out-of-possession teams.

    Parameters
    ----------
    frame_df : pd.DataFrame
        Data for one frame (all players). Must contain ``player_id``,
        ``team_id_opta``, ``x``, ``y``, and ``team_in_possession``.
    team_in_possession_col : str
        Column indicating which team has the ball.
    team_id_col : str
        Column for team ID.

    Returns
    -------
    dict[int, dict[int, float]]
        Nested dict: ``{team_id: {player_id: distance_to_centroid}}``.
    """
    ball_mask = frame_df["player_id"] == _BALL_PLAYER_ID
    if ball_mask.any():
        in_poss = frame_df.loc[ball_mask, team_in_possession_col].iloc[0]
    else:
        in_poss = np.nan

    outfield = frame_df[~ball_mask].copy()
    if len(outfield) == 0:
        return {}

    result: dict[int, dict[int, float]] = {}

    for team_id in outfield[team_id_col].unique():
        team_players = outfield[outfield[team_id_col] == team_id]

        # Only out-of-possession teams
        if not np.isnan(in_poss) and int(team_id) == int(in_poss):
            result[int(team_id)] = {}
            continue

        x = team_players["x"].values.astype(np.float64)
        y = team_players["y"].values.astype(np.float64)

        if len(x) == 0:
            result[int(team_id)] = {}
            continue

        # Team centroid (mean x, mean y)
        cx = float(np.mean(x))
        cy = float(np.mean(y))

        # Distance of each player to centroid
        dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        team_map: dict[int, float] = {}
        for i, (_, row) in enumerate(team_players.iterrows()):
            pid = int(row["player_id"])
            team_map[pid] = float(dists[i])

        result[int(team_id)] = team_map

    return result


def compute_centroid_distance_block(
    match_df: pd.DataFrame,
    block: dict[str, Any],
    team_in_possession_col: str = "team_in_possession",
    team_id_col: str = "team_id_opta",
) -> list[dict[str, Any]]:
    """Aggregate centroid distances for one block across all frames.

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
        Records ready for the output DataFrame, one per player per block.
    """
    start = block["start_frame"]
    end = block["end_frame"]

    frame_mask = match_df["frame_count"].between(start, end, inclusive="left")
    block_df = match_df[frame_mask]

    if len(block_df) == 0:
        return []

    # Accumulate per-player distances across frames
    player_distances: dict[tuple[int, int], list[float]] = {}  # (player_id, team_id) → [dist]

    for frame_val in sorted(block_df["frame_count"].unique()):
        fdf = block_df[block_df["frame_count"] == frame_val]
        team_map = compute_centroid_distance_frame(
            fdf,
            team_in_possession_col=team_in_possession_col,
            team_id_col=team_id_col,
        )
        for team_id, player_dists in team_map.items():
            for pid, dist in player_dists.items():
                # NaN means in-possession (should not happen since we skip in-possession)
                if not np.isnan(dist):
                    player_distances.setdefault((pid, int(team_id)), []).append(dist)

    # Aggregate per player for this block
    records: list[dict[str, Any]] = []
    n_frames_in_block = block_df["frame_count"].nunique()

    for (pid, team_id), dist_values in player_distances.items():
        if len(dist_values) == 0:
            continue
        mean_dist = float(np.mean(dist_values))
        n_frames = len(dist_values)
        records.append({
            "block_id": block["block_id"],
            "phase": block["phase"],
            "player_id": pid,
            "team_id_opta": team_id,
            "signal_name": _SIGNAL_NAME,
            "signal_value": round(mean_dist, 6),
            "n_frames": n_frames,
        })

    return records


# ═══════════════════════════════════════════════════════════════════════════
# Block Conversion Helper
# ═══════════════════════════════════════════════════════════════════════════

def _blocks_to_dicts(blocks: list[Any]) -> list[dict[str, Any]]:
    """Convert blocks (list[DataFrame] or list[dict]) to list[dict].

    Normalises the block format for consistent frame-range filtering.
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
class TeamCentroidDistanceSignal(SignalBase):
    """Team centroid distance signal — per-player distance to team centroid.

    For each frame during out-of-possession phases, computes the Euclidean
    distance from each player to their team's centroid (mean x, mean y).
    Averages over each 5-minute block per player.

    Larger values indicate a player is further from the team's collective
    shape, potentially signalling degraded positional discipline.
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
    ) -> pd.DataFrame:
        """Compute team centroid distance per block per player.

        Parameters
        ----------
        match_df : pd.DataFrame
            Full tracking data for one match. Must contain columns:
            ``player_id``, ``team_id_opta``, ``frame_count``, ``x``, ``y``,
            ``team_in_possession``.
        blocks : list[DataFrame | dict]
            Block definitions from ``split_into_blocks``.
        game_id : str
            Match identifier.
        team_in_possession_col : str
            Column indicating which team has the ball.
        team_id_col : str
            Column for team ID.

        Returns
        -------
        pd.DataFrame
            Standard signal output with one row per player per block.
        """
        if len(match_df) == 0 or len(blocks) == 0:
            empty = pd.DataFrame(columns=[
                "game_id", "block_id", "phase", "player_id", "team_id_opta",
                "signal_name", "signal_value", "n_frames",
            ])
            return empty

        # Normalise blocks to dicts
        block_dicts = _blocks_to_dicts(blocks)

        # Compute centroid distances per block
        all_records: list[dict[str, Any]] = []
        for bd in block_dicts:
            records = compute_centroid_distance_block(
                match_df, bd,
                team_in_possession_col=team_in_possession_col,
                team_id_col=team_id_col,
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
        """Validate centroid distance output.

        Checks:
        - Standard schema compliance
        - signal_value (mean distance) non-negative
        - n_frames non-negative
        - player_id values are non-negative (not -1/ball)
        """
        super().validate(output_df)
        if len(output_df) == 0:
            return True

        # Non-negative distances
        sv = output_df["signal_value"]
        if sv.min() < 0.0:
            raise ValueError(
                f"signal_value (mean centroid distance) contains negative "
                f"values (min={sv.min():.4f})"
            )

        # n_frames non-negative
        if output_df["n_frames"].min() < 0:
            raise ValueError(
                f"n_frames contains negative values "
                f"(min={output_df['n_frames'].min()})"
            )

        # player_id should not be ball (-1)
        bad_players = output_df[output_df["player_id"] == _BALL_PLAYER_ID]
        if len(bad_players) > 0:
            raise ValueError(
                f"Found {len(bad_players)} rows with player_id={_BALL_PLAYER_ID} "
                f"(ball player). Ball should not have a centroid distance."
            )

        # Sanity: distances shouldn't exceed pitch diagonal (~140m)
        max_dist = sv.max()
        if max_dist > 150.0:
            self.logger.warning(
                "max centroid distance=%.1f m exceeds pitch diagonal (~140m); "
                "verify coordinate system", max_dist
            )

        return True
