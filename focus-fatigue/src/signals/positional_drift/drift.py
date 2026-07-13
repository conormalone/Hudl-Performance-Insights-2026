"""Positional drift computation.

Measures how far a player's actual position is from their role-based
expected position (from the shape.json model) during out-of-possession
phases.

The core hypothesis: as defenders experience cognitive fatigue, their
spatial awareness degrades, causing them to drift further from their
expected role positions. This drift manifests as a larger Euclidean
distance between actual position and the shape-model centroid.

The pipeline:
1.  Use :func:`bridge.vectorise_expected_positions` to add expected
    position columns to the tracking DataFrame.
2.  Compute per-frame drift as the Euclidean distance between actual
    and expected position (out-of-possession only).
3.  Aggregate per-block, per-player statistics (mean, P90, max).

Direction of Play (DOP)
-----------------------
Shape.json positions are generally DOP-normalised so that positive x
is the attacking direction for both teams. The bridge functions in
:mod:`bridge` handle this normalisation. The drift computation assumes
all coordinates are already in a consistent DOP-normalised frame.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .bridge import vectorise_expected_positions
from .config import DriftConfig


# ═══════════════════════════════════════════════════════════════════════════
# Drift Computation
# ═══════════════════════════════════════════════════════════════════════════


def compute_drift(
    df: pd.DataFrame,
    player_role_map: dict[int, dict[int, dict[str, Any]]],
    config: DriftConfig,
    *,
    team_in_possession_col: str = "team_in_possession",
    team_id_col: str = "team_id_opta",
) -> pd.DataFrame:
    """Compute positional drift per frame per player.

    Drift is defined as the Euclidean distance between the player's
    actual position and their shape-model expected position:

        drift = sqrt((x - expected_x)² + (y - expected_y)²)

    The computation is restricted to **out-of-possession** frames,
    i.e. when the player's team does **not** have the ball. When
    ``team_in_possession`` is NaN (ball not in play), drift is also
    set to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Tracking DataFrame. Must contain columns:
        ``frame``, ``player_id``, ``team_id_opta``, ``x``, ``y``,
        and the DOP-normalised possession indicator.
    player_role_map : dict
        Player-to-role mapping from
        :func:`bridge.build_player_role_map`.
    config : DriftConfig
        Configuration controlling FPS, min fit score, etc.
    team_in_possession_col : str
        Column name indicating which team has possession. NaN/None
        values mean the ball is out of play.
    team_id_col : str
        Column name for the player's team Opta ID.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with the following columns added
        (or overwritten):

        - **expected_x** (*float64*) — Shape-model expected x-coordinate.
          NaN when no shape entry is available.
        - **expected_y** (*float64*) — Shape-model expected y-coordinate.
          NaN when no shape entry is available.
        - **fit_score** (*float64*) — Quality of the shape entry used.
          NaN when unavailable.
        - **shape_role** (*str*) — Role label from the shape model
          (e.g. ``"CB"``, ``"RB"``).
        - **drift_m** (*float64*) — Euclidean drift distance in metres.
          NaN for in-possession or out-of-play frames.

    Notes
    -----
    This function does **not** filter out-of-possession frames. It
    computes the drift column for all rows, then masks out
    in-possession and out-of-play frames by setting ``drift_m`` to NaN.
    Downstream aggregations should filter on non-NaN drift values.
    """
    fps = config.frames_per_second
    frames_per_minute = fps * 60

    # ── Step 1: Vectorise expected positions ───────────────────────────
    result = vectorise_expected_positions(
        df=df,
        player_role_map=player_role_map,
        frames_per_minute=frames_per_minute,
        allow_forward_fill=True,
    )

    # ── Step 2: Compute Euclidean drift ────────────────────────────────
    has_expected = (
        result["expected_x"].notna() & result["expected_y"].notna()
    )

    result["drift_m"] = np.where(
        has_expected,
        np.sqrt(
            (result["x"] - result["expected_x"]) ** 2
            + (result["y"] - result["expected_y"]) ** 2
        ),
        np.nan,
    )

    # ── Step 3: Mask out-of-possession only ────────────────────────────
    # Out-of-possession: player's team does NOT have the ball
    # team_in_possession is NaN → ball out of play → drift is NaN
    if team_in_possession_col in result.columns and team_id_col in result.columns:
        own_team = result[team_id_col]
        in_poss_team = result[team_in_possession_col]

        # Where team_in_possession is NaN, drift is NaN (out of play)
        ball_in_play = in_poss_team.notna()
        out_of_possession = ball_in_play & (own_team != in_poss_team)

        result["drift_m"] = result["drift_m"].where(out_of_possession, np.nan)
    else:
        # No possession info — don't mask (warn but continue)
        pass

    # ── Step 4: Clamp implausible values ────────────────────────────────
    max_drift = config.max_plausible_drift_m
    result.loc[result["drift_m"] > max_drift, "drift_m"] = np.nan

    # ── Step 5: Drop expected positions for rows with low fit score ────
    min_fit = config.min_fit_score
    low_fit = result["fit_score"].notna() & (result["fit_score"] < min_fit)
    result.loc[low_fit, "drift_m"] = np.nan

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Block Aggregation
# ═══════════════════════════════════════════════════════════════════════════


def aggregate_drift_by_block(
    df: pd.DataFrame,
    blocks: list[dict[str, Any]],
    config: DriftConfig,
    *,
    game_id: str = "",
    valid_only: bool = True,
) -> pd.DataFrame:
    """Aggregate drift statistics per block and per player.

    Produces a standardised output DataFrame with one row per
    (player, block) combination.

    Parameters
    ----------
    df : pd.DataFrame
        Tracking DataFrame that has been processed through
        :func:`compute_drift` (or is a subset with drift columns).
        Must contain ``player_id``, ``team_id_opta``, ``drift_m``,
        ``fit_score``, and ``frame`` columns.
    blocks : list[dict[str, Any]]
        Block definitions. Each block is a dict with:
        - ``block_id`` (str) — e.g. ``"block_0"``
        - ``phase`` (int) — match period (1/2)
        - ``start_frame`` (int)
        - ``end_frame`` (int)
    config : DriftConfig
        Configuration (used for fit-score threshold reference in output).
    game_id : str
        Optional match identifier for the output.
    valid_only : bool
        If ``True`` (default), only rows with non-NaN ``drift_m`` are
        used in aggregation. Set to ``False`` to still produce rows
        even when no valid drift data exists (with NaN signal values).

    Returns
    -------
    pd.DataFrame
        A DataFrame conforming to the standard signal output schema
        with columns:

        - **game_id** (*str*) — Match identifier.
        - **block_id** (*str*) — Block identifier.
        - **phase** (*int*) — Match period.
        - **player_id** (*int*) — Player identifier.
        - **team_id_opta** (*int*) — Team Opta ID.
        - **signal_name** (*str*) — Always ``"positional_drift"``.
        - **signal_value** (*float*) — Mean drift distance (metres)
          for this player in this block.
        - **n_frames** (*int*) — Number of frames with valid drift
          data for this player in this block.

        Additionally includes these extra columns:

        - **drift_p90** (*float*) — 90th percentile drift distance.
        - **drift_max** (*float*) — Maximum drift distance.
        - **drift_std** (*float*) — Standard deviation of drift.
        - **mean_fit_score** (*float*) — Mean shape fit score across
          frames used.
    """
    records: list[dict[str, Any]] = []

    for b, block in enumerate(blocks):
        block_id = block.get("block_id", f"block_{b}")
        phase = block.get("phase", 1)
        start = block.get("start_frame", 0)
        end = block.get("end_frame", 0)

        # Slice to block frames
        block_mask = df["frame"].between(start, end, inclusive="left")

        if not block_mask.any():
            continue

        block_df = df.loc[block_mask]

        # Get players who were on the pitch in this block
        players_in_block = (
            block_df[["player_id", "team_id_opta"]]
            .drop_duplicates(subset="player_id")
            .dropna(subset=["player_id"])
        )

        for _, player_row in players_in_block.iterrows():
            pid = int(player_row["player_id"])
            team_opta = int(player_row["team_id_opta"])

            # Filter to this player's frames
            player_mask = block_df["player_id"] == pid
            player_df = block_df.loc[player_mask]

            # Extract valid drift values
            drift_values = player_df["drift_m"].dropna()

            n_valid = len(drift_values)

            if n_valid == 0 and valid_only:
                continue

            # Compute statistics
            if n_valid > 0:
                mean_drift = float(drift_values.mean())
                p90_drift = float(drift_values.quantile(0.90))
                max_drift = float(drift_values.max())
                std_drift = float(drift_values.std())
            else:
                mean_drift = np.nan
                p90_drift = np.nan
                max_drift = np.nan
                std_drift = np.nan

            # Mean fit score across all frames (not just valid drift)
            fit_scores = player_df["fit_score"].dropna()
            mean_fit = float(fit_scores.mean()) if len(fit_scores) > 0 else np.nan

            records.append(
                {
                    # Standard schema columns
                    "game_id": game_id,
                    "block_id": block_id,
                    "phase": phase,
                    "player_id": pid,
                    "team_id_opta": team_opta,
                    "signal_name": "positional_drift",
                    "signal_value": mean_drift,
                    "n_frames": n_valid,
                    # Extra columns
                    "drift_p90": p90_drift,
                    "drift_max": max_drift,
                    "drift_std": std_drift,
                    "mean_fit_score": mean_fit,
                }
            )

    output_df = pd.DataFrame(records)

    if len(output_df) == 0:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(
            columns=[
                "game_id",
                "block_id",
                "phase",
                "player_id",
                "team_id_opta",
                "signal_name",
                "signal_value",
                "n_frames",
                "drift_p90",
                "drift_max",
                "drift_std",
                "mean_fit_score",
            ]
        )

    # Standardise types
    output_df["player_id"] = output_df["player_id"].astype(int)
    output_df["team_id_opta"] = output_df["team_id_opta"].astype(int)
    output_df["phase"] = output_df["phase"].astype(int)
    output_df["n_frames"] = output_df["n_frames"].astype(int)
    output_df["game_id"] = output_df["game_id"].astype(str)
    output_df["block_id"] = output_df["block_id"].astype(str)
    output_df["signal_name"] = output_df["signal_name"].astype(str)

    return output_df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: Full Drift Pipeline
# ═══════════════════════════════════════════════════════════════════════════


def compute_drift_pipeline(
    tracking_df: pd.DataFrame,
    player_role_map: dict[int, dict[int, dict[str, Any]]],
    blocks: list[dict[str, Any]],
    config: DriftConfig,
    *,
    game_id: str = "",
    team_in_possession_col: str = "team_in_possession",
    team_id_col: str = "team_id_opta",
) -> pd.DataFrame:
    """Compute per-frame drift and aggregate to per-block statistics.

    A convenience wrapper that calls :func:`compute_drift` followed by
    :func:`aggregate_drift_by_block` in a single pass.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Full tracking data for a single match.
    player_role_map : dict
        Player-to-role mapping from :func:`bridge.build_player_role_map`.
    blocks : list[dict[str, Any]]
        Block definitions (start/end frames).
    config : DriftConfig
        Drift computation configuration.
    game_id : str
        Optional match identifier.
    team_in_possession_col : str
        Column for DOP-normalised possession indicator.
    team_id_col : str
        Column for the player's team Opta ID.

    Returns
    -------
    pd.DataFrame
        Aggregated per-block, per-player drift statistics.
    """
    # Compute per-frame drift
    drift_df = compute_drift(
        df=tracking_df,
        player_role_map=player_role_map,
        config=config,
        team_in_possession_col=team_in_possession_col,
        team_id_col=team_id_col,
    )

    # Aggregate to blocks
    output_df = aggregate_drift_by_block(
        df=drift_df,
        blocks=blocks,
        config=config,
        game_id=game_id,
    )

    return output_df
