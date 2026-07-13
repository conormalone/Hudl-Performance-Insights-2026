"""Signal 5 — Transition Recognition.

Measures how quickly defenders recognise and react to possession
transitions (turnovers). The core hypothesis is that cognitively
fatigued defenders exhibit longer perception-reaction delays while
their physical sprint speed remains intact.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..base import SignalBase
from ..config import SignalConfig, DEFAULT_SIGNAL_CONFIG
from ..registry import register_signal
from .config import TransitionConfig, DEFAULT_TRANSITION_CONFIG
from .detector import detect_transitions, classify_transition_type
from .latency import compute_reaction_time, aggregate_latency_by_block

# ── Public API ──────────────────────────────────────────────────────────────

__all__ = [
    # Config
    "TransitionConfig",
    "DEFAULT_TRANSITION_CONFIG",
    # Detection
    "detect_transitions",
    "classify_transition_type",
    # Latency
    "compute_reaction_time",
    "aggregate_latency_by_block",
    # Class
    "TransitionRecognitionSignal",
]


# ═══════════════════════════════════════════════════════════════════════════
# Registered Signal Class
# ═══════════════════════════════════════════════════════════════════════════


@register_signal
class TransitionRecognitionSignal(SignalBase):
    """Signal 5 — Transition Recognition.

    Computes per-defender reaction latency to possession-change events.
    A longer reaction latency indicates slower cognitive recognition of
    the transition (fatigue indicator).

    The pipeline runs:
    1. :func:`detect_transitions` — identify possession flip frames
    2. :func:`classify_transition_type` — label as expected / surprise
    3. :func:`compute_reaction_time` — measure per-defender latency
    4. :func:`aggregate_latency_by_block` — summarise into per-block values
    """

    signal_name = "transition_latency"
    """Registered signal name used in the output schema and registry."""

    def __init__(
        self,
        signal_config: SignalConfig | None = None,
        transition_config: TransitionConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(config=signal_config, logger=logger)
        self.transition_config = transition_config or DEFAULT_TRANSITION_CONFIG

    # ── Public API ─────────────────────────────────────────────────────────

    def compute(
        self,
        match_df: pd.DataFrame,
        blocks: list[dict[str, Any]],
        *,
        game_id: str = "",
        own_goal_direction: str = "left",
        team_in_possession_col: str = "team_in_possession",
    ) -> pd.DataFrame:
        """Compute transition reaction latency for all players across blocks.

        Parameters
        ----------
        match_df : pd.DataFrame
            Full tracking data for a single match. Must contain:
            ``frame_count``, ``player_id``, ``team_id_opta``,
            ``team_in_possession``, ``vx_smooth`` / ``vy_smooth``,
            ``v_mag``, ``heading``, and positional coordinates.
        blocks : list[dict[str, Any]]
            Block definitions (see :class:`SignalBase.compute`).
        game_id : str
            Optional match identifier for the output.
        own_goal_direction : str
            DOP-normalised direction of own goal (``'left'`` or ``'right'``).
        team_in_possession_col : str
            Column name indicating which team has the ball.

        Returns
        -------
        pd.DataFrame
            Standardised signal output with one row per player-block
            combination and ``signal_value`` set to the mean reaction
            latency (seconds).
        """
        cfg = self.transition_config
        fps = cfg.frames_per_second

        # Step 1: Detect transitions
        self.logger.info("Detecting possession transitions...")
        trans_df = detect_transitions(
            df=match_df,
            team_in_possession_col=team_in_possession_col,
            min_gap_frames=cfg.min_gap_frames,
            config=cfg,
        )

        if len(trans_df) == 0:
            self.logger.warning("No transitions detected — returning empty result.")
            return pd.DataFrame(
                columns=[
                    "game_id", "block_id", "phase", "player_id",
                    "team_id_opta", "signal_name", "signal_value", "n_frames",
                ]
            )

        self.logger.info("Found %d transition events.", len(trans_df))

        # Step 2: Classify transition types
        self.logger.info("Classifying transition types (expected / surprise)...")
        trans_df = classify_transition_type(
            df=match_df,
            trans_df=trans_df,
            config=cfg,
        )

        # Step 3: Compute reaction time per defender per transition
        self.logger.info("Computing defender reaction latencies...")
        latency_df = compute_reaction_time(
            df=match_df,
            trans_df=trans_df,
            config=cfg,
            own_goal_direction=own_goal_direction,
        )

        # Carry forward transition metadata (frame, transition_type) for
        # block aggregation
        trans_meta = trans_df[["transition_id", "frame", "transition_type"]].copy()
        trans_meta["frame"] = trans_meta["frame"].astype(int)
        latency_df = latency_df.merge(trans_meta, on="transition_id", how="left")

        # Step 4: Aggregate into per-block, per-player signal values
        self.logger.info("Aggregating latencies into blocks...")
        output_df = aggregate_latency_by_block(
            latency_df=latency_df,
            blocks=blocks,
            config=cfg,
            game_id=game_id,
        )

        self.logger.info(
            "Computed transition_latency signal: %d rows.",
            len(output_df),
        )
        return output_df

    def validate(self, output_df: pd.DataFrame) -> bool:
        """Validate the transition latency output.

        In addition to the standard schema checks, verifies that:
        - ``signal_value`` is non-negative (reaction time cannot be negative)
        - Values are within a plausible range (< 5.0 seconds)

        Parameters
        ----------
        output_df : pd.DataFrame
            Output from :meth:`compute`.

        Returns
        -------
        bool
            ``True`` if all checks pass.

        Raises
        ------
        ValueError
            Describing the first validation failure.
        """
        # Run standard schema validation first
        super().validate(output_df)

        if len(output_df) == 0:
            return True

        # Signal-specific checks
        sv = output_df["signal_value"]
        if sv.min() < 0:
            raise ValueError(
                f"signal_value contains negative values "
                f"(min={sv.min():.3f}). Reaction time cannot be negative."
            )

        max_allowed = self.transition_config.max_reaction_time_s
        if sv.max() > max_allowed:
            raise ValueError(
                f"signal_value exceeds max_reaction_time_s={max_allowed:.1f} s: "
                f"max observed = {sv.max():.3f} s."
            )

        return True
