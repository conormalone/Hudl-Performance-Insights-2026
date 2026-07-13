"""Signal 2 — Shift Latency.

Measures how quickly defenders react to sudden 'shifts' in play:
ball speed spikes (fast passes/long balls) and aggressive opponent runs.
The core hypothesis is that cognitively fatigued defenders exhibit
slower reaction times to these sudden events while their physical
speed remains intact.

The pipeline runs:
1. :func:`detect_ball_speed_spikes` — identify ball speed spike frames
2. :func:`detect_opponent_runs` — identify aggressive attacker runs
3. :func:`compute_shift_reaction_time` — measure per-defender latency
4. :func:`aggregate_shift_latency_by_block` — summarise into per-block values
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..base import SignalBase
from ..config import SignalConfig, DEFAULT_SIGNAL_CONFIG
from ..registry import register_signal
from .config import ShiftLatencyConfig, DEFAULT_SHIFT_LATENCY_CONFIG
from .triggers import detect_ball_speed_spikes, detect_opponent_runs, detect_all_triggers
from .latency import compute_shift_reaction_time, aggregate_shift_latency_by_block

# ── Public API ──────────────────────────────────────────────────────────────

__all__ = [
    # Config
    "ShiftLatencyConfig",
    "DEFAULT_SHIFT_LATENCY_CONFIG",
    # Triggers
    "detect_ball_speed_spikes",
    "detect_opponent_runs",
    "detect_all_triggers",
    # Latency
    "compute_shift_reaction_time",
    "aggregate_shift_latency_by_block",
    # Class
    "ShiftLatencySignal",
]


# ═══════════════════════════════════════════════════════════════════════════
# Registered Signal Class
# ═══════════════════════════════════════════════════════════════════════════


@register_signal
class ShiftLatencySignal(SignalBase):
    """Signal 2 — Shift Latency.

    Computes per-defender reaction latency to ball speed spikes and
    opponent runs. A longer reaction latency indicates slower cognitive
    recognition of sudden play shifts (fatigue indicator).

    The pipeline runs:
    1. Detect ball speed spikes and opponent runs
    2. Measure per-defender reaction latency for each trigger
    3. Aggregate into per-block, per-player signal values
    """

    signal_name = "shift_latency"
    """Registered signal name used in the output schema and registry."""

    def __init__(
        self,
        signal_config: SignalConfig | None = None,
        shift_config: ShiftLatencyConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(config=signal_config, logger=logger)
        self.shift_config = shift_config or DEFAULT_SHIFT_LATENCY_CONFIG

    # ── Public API ─────────────────────────────────────────────────────────

    def compute(
        self,
        match_df: pd.DataFrame,
        blocks: list[dict[str, Any]],
        *,
        game_id: str = "",
        own_goal_direction: str = "left",
    ) -> pd.DataFrame:
        """Compute shift reaction latency for all players across blocks.

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

        Returns
        -------
        pd.DataFrame
            Standardised signal output with one row per player-block
            combination and ``signal_value`` set to the mean reaction
            latency (seconds).
        """
        cfg = self.shift_config

        # Step 1: Detect all triggers
        self.logger.info("Detecting ball speed spikes...")
        spike_df = detect_ball_speed_spikes(match_df, cfg)
        self.logger.info("  Found %d ball speed spikes.", len(spike_df))

        self.logger.info("Detecting opponent runs...")
        run_df = detect_opponent_runs(match_df, cfg)
        self.logger.info("  Found %d opponent runs.", len(run_df))

        trigger_df = detect_all_triggers(match_df, cfg)
        n_triggers = len(trigger_df)
        self.logger.info("Total trigger events: %d", n_triggers)

        if n_triggers == 0:
            self.logger.warning("No triggers detected — returning empty result.")
            return pd.DataFrame(
                columns=[
                    "game_id", "block_id", "phase", "player_id",
                    "team_id_opta", "signal_name", "signal_value", "n_frames",
                ]
            )

        # Step 2: Compute reaction time per defender per trigger
        self.logger.info("Computing defender shift reaction latencies...")
        latency_df = compute_shift_reaction_time(
            df=match_df,
            trigger_df=trigger_df,
            config=cfg,
            own_goal_direction=own_goal_direction,
        )

        n_valid = latency_df["valid"].sum() if "valid" in latency_df.columns else 0
        self.logger.info("  Valid reactions: %d / %d", n_valid, len(latency_df))

        # Step 3: Aggregate into per-block, per-player signal values
        self.logger.info("Aggregating shift latencies into blocks...")
        output_df = aggregate_shift_latency_by_block(
            latency_df=latency_df,
            blocks=blocks,
            config=cfg,
            game_id=game_id,
        )

        self.logger.info(
            "Computed shift_latency signal: %d rows.", len(output_df)
        )
        return output_df

    def validate(self, output_df: pd.DataFrame) -> bool:
        """Validate the shift latency output.

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
        super().validate(output_df)

        if len(output_df) == 0:
            return True

        sv = output_df["signal_value"]
        if sv.min() < 0:
            raise ValueError(
                f"signal_value contains negative values "
                f"(min={sv.min():.3f}). Reaction time cannot be negative."
            )

        max_allowed = self.shift_config.max_reaction_time_s
        if sv.max() > max_allowed:
            raise ValueError(
                f"signal_value exceeds max_reaction_time_s={max_allowed:.1f} s: "
                f"max observed = {sv.max():.3f} s."
            )

        return True
