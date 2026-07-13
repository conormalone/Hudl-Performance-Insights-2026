"""Pressing accuracy classification and block aggregation for Signal 3.

Classifies individual pressing events as *correct* or *wasteful* based
on the defender's intercept probability, then aggregates these into
per-block, per-player summary statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PressingConfig

# ── Public Functions ───────────────────────────────────────────────────────


def classify_pressing_accuracy(
    df: pd.DataFrame,
    tti_df: pd.DataFrame,
    pressing_df: pd.DataFrame,
    config: PressingConfig,
) -> pd.DataFrame:
    """Classify each pressing event as 'correct' or 'wasteful'.

    A press is classified as:

    - **correct** if the defender's ``intercept_probability`` exceeds
      ``config.correct_press_threshold`` (default > 0.3).
    - **wasteful** if the intercept probability is at or below the
      threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Full tracking data (used here only for schema consistency with
        other pipeline stages; not directly required for classification).
    tti_df : pd.DataFrame
        Output of :func:`src.signals.pressing.tti.compute_tti` with
        per-frame per-defender TTI estimates.
    pressing_df : pd.DataFrame
        Output of :func:`src.signals.pressing.detection.detect_pressing_events`,
        containing the ``is_pressing`` boolean column.
    config : PressingConfig
        Configuration containing ``correct_press_threshold``.

    Returns
    -------
    pd.DataFrame
        The input ``pressing_df`` with two additional columns:

        - ``is_correct_press`` (bool): ``True`` when the press is
          classified as correct (only meaningful when ``is_pressing``).
        - ``press_quality`` (str): ``"correct"``, ``"wasteful"``, or
          ``"none"`` (when not pressing).
    """
    _ = df  # accepted for pipeline consistency; not used directly

    result = pressing_df.copy()

    # Only classify frames where the defender is actively pressing
    result["is_correct_press"] = (
        result["is_pressing"]
        & (result["intercept_probability"] > config.correct_press_threshold)
    )

    # Human-readable label
    conditions = [
        result["is_correct_press"],
        result["is_pressing"] & ~result["is_correct_press"],
    ]
    choices = ["correct", "wasteful"]
    result["press_quality"] = np.select(conditions, choices, default="none")

    return result


def aggregate_pressing_by_block(
    df: pd.DataFrame,
    blocks: list[pd.DataFrame],
    config: PressingConfig,
    *,
    game_id: str = "",
) -> pd.DataFrame:
    """Aggregate pressing accuracy statistics to per-block, per-player.

    Computes the following summary metrics for each player-block
    combination:

    - ``pressing_accuracy``: fraction of presses that were correct
      (correct / total), set to 0.0 when there are no presses.
    - ``n_presses``: total number of frames where the defender was
      pressing.
    - ``mean_intercept_prob``: average intercept probability across all
      pressing frames.
    - ``p90_tti``: 90th percentile of TTI values across pressing frames
      (higher = worst-case pressing position).

    **Output schema:** The returned DataFrame conforms to the standard
    ``OUTPUT_COLUMNS`` format defined in :mod:`src.signals.output_schema`,
    with ``signal_value`` set to ``pressing_accuracy``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`classify_pressing_accuracy`, containing columns:
        ``frame_count``, ``player_id``, ``is_pressing``,
        ``is_correct_press``, ``intercept_probability``, ``tti_value``.
    blocks : list of pd.DataFrame
        Block-segmented tracking data from :func:`split_into_blocks`.
        Each DataFrame has a ``block_id`` column and ``frame_count``.
    config : PressingConfig
        Configuration (used for frame rate reference in future
        time-based metrics; reserved for consistency).
    game_id : str, optional
        Match identifier for the output DataFrame.

    Returns
    -------
    pd.DataFrame
        Standardised signal output with columns:
        ``game_id``, ``block_id``, ``phase``, ``player_id``,
        ``team_id_opta``, ``signal_name``, ``signal_value``,
        ``n_frames``.

        Also includes extended columns:
        ``n_presses``, ``mean_intercept_prob``, ``p90_tti``,
        ``total_correct``, ``total_wasteful``.
    """
    _ = config  # reserved for future time-based extensions

    required_cols = [
        "frame_count", "player_id", "is_pressing",
        "is_correct_press", "intercept_probability", "tti_value",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame missing required columns: {missing}"
        )

    # ── Build block membership lookup from DataFrame blocks ────────────
    block_records: list[dict] = []
    for block in blocks:
        bid = str(block["block_id"].iloc[0])
        phase = int(bid.split("_")[0])
        start_frame = int(block["frame_count"].min())
        end_frame = int(block["frame_count"].max())
        block_records.append({
            "block_id": bid,
            "phase": phase,
            "start_frame": start_frame,
            "end_frame": end_frame,
        })

    block_df = pd.DataFrame(block_records)

    # Cross-join blocks with pressing data to assign each frame to its block
    df["_key"] = 1
    block_df["_key"] = 1
    merged = df.merge(block_df, on="_key")
    merged = merged.drop(columns="_key")

    # Filter to frames within each block's range
    in_block = (
        (merged["frame_count"] >= merged["start_frame"])
        & (merged["frame_count"] < merged["end_frame"])
    )
    merged = merged[in_block].copy()

    # ── Aggregate per (block, player) ──────────────────────────────────
    grouped = merged.groupby(
        ["block_id", "phase", "player_id"], as_index=False
    )

    agg = grouped.agg(
        n_frames=("frame_count", "nunique"),
        n_presses=("is_pressing", "sum"),
        correct_presses=("is_correct_press", "sum"),
        mean_intercept_prob=("intercept_probability", "mean"),
        p90_tti=("tti_value", lambda x: x.quantile(0.90)),
    )

    # Compute accuracy
    agg["pressing_accuracy"] = np.where(
        agg["n_presses"] > 0,
        agg["correct_presses"] / agg["n_presses"],
        0.0,
    )

    agg["total_correct"] = agg["correct_presses"].astype(int)
    agg["total_wasteful"] = (agg["n_presses"] - agg["correct_presses"]).astype(int)

    # ── Standardise output columns ─────────────────────────────────────
    # Carry team_id from the first available row in each group
    team_map = df[["player_id", "team_id_opta"]].drop_duplicates("player_id")
    if "team_id_opta" not in df.columns:
        # Attempt to find it from original data
        team_map = team_map if "team_id_opta" in team_map.columns else None

    output = agg.merge(team_map, on="player_id", how="left") if team_map is not None else agg.copy()

    if "team_id_opta" not in output.columns:
        output["team_id_opta"] = 0

    output["game_id"] = game_id
    output["signal_name"] = "pressing_accuracy"
    output["signal_value"] = output["pressing_accuracy"].astype(float)

    # ── Standard column order ──────────────────────────────────────────
    standard_cols = [
        "game_id", "block_id", "phase", "player_id", "team_id_opta",
        "signal_name", "signal_value", "n_frames",
    ]
    extended_cols = [
        "n_presses", "mean_intercept_prob", "p90_tti",
        "total_correct", "total_wasteful", "pressing_accuracy",
    ]

    # Keep only standard cols plus extended cols that exist
    all_cols = standard_cols + [c for c in extended_cols if c in output.columns]

    return output[all_cols].reset_index(drop=True)
