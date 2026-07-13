"""Signal 3 — Pressing Accuracy (Bekkers Time-To-Intercept).

Measures how effectively defenders press opposition attackers using
the Bekkers Time-To-Intercept (TTI) framework. The core hypothesis is
that mentally fatigued defenders exhibit less accurate pressing —
either pressing when they cannot realistically intercept (wasteful)
or failing to press when they have a strong intercept opportunity
(missed).

The pipeline runs:
1. :func:`compute_tti` — estimate intercept time for all defender-attacker pairs
2. :func:`detect_pressing_events` — identify frames with active pressing
3. :func:`classify_pressing_accuracy` — label presses as correct or wasteful
4. :func:`aggregate_pressing_by_block` — summarise into per-block signal values
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..base import SignalBase
from ..config import SignalConfig, DEFAULT_SIGNAL_CONFIG
from ..registry import register_signal
from .config import PressingConfig, DEFAULT_PRESSING_CONFIG
from .tti import compute_tti, compute_tta_threshold
from .detection import detect_pressing_events
from .accuracy import classify_pressing_accuracy, aggregate_pressing_by_block

# ── Public API ─────────────────────────────────────────────────────────────

__all__ = [
    # Config
    "PressingConfig",
    "DEFAULT_PRESSING_CONFIG",
    # TTI
    "compute_tti",
    "compute_tta_threshold",
    # Detection
    "detect_pressing_events",
    # Accuracy
    "classify_pressing_accuracy",
    "aggregate_pressing_by_block",
    # Class
    "PressingAccuracySignal",
]


# ═══════════════════════════════════════════════════════════════════════════
# Registered Signal Class
# ═══════════════════════════════════════════════════════════════════════════


@register_signal
class PressingAccuracySignal(SignalBase):
    """Signal 3 — Pressing Accuracy via Bekkers Time-To-Intercept.

    Computes per-defender pressing accuracy across match blocks using
    the TTI framework. ``signal_value`` is set to the fraction of
    pressing actions classified as correct for a player-block combination.

    A lower pressing accuracy indicates that the defender is pressing
    when they have little realistic chance of intercepting — a potential
    sign of impaired decision-making (cognitive fatigue).
    """

    signal_name = "pressing_accuracy"
    """Registered signal name used in the output schema and registry."""

    def __init__(
        self,
        signal_config: SignalConfig | None = None,
        pressing_config: PressingConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(config=signal_config, logger=logger)
        self.pressing_config = pressing_config or DEFAULT_PRESSING_CONFIG

    # ── Public API ─────────────────────────────────────────────────────────

    def compute(
        self,
        match_df: pd.DataFrame,
        blocks: list[dict[str, Any]],
        *,
        game_id: str = "",
        own_team_id: int = 0,
        opponent_team_id: int = 0,
    ) -> pd.DataFrame:
        """Compute pressing accuracy across all blocks.

        Parameters
        ----------
        match_df : pd.DataFrame
            Full tracking data for a single match. Must contain:
            ``frame_count``, ``player_id``, ``team_id_opta``,
            ``x``, ``y``, ``vx_smooth``, ``vy_smooth``.
            Optionally ``goalkeeper`` (bool) or ``jersey_number`` (int).
        blocks : list[dict[str, Any]]
            Block definitions (see :class:`SignalBase.compute`).
        game_id : str, optional
            Match identifier for the output.
        own_team_id : int
            Team ID of the defending / pressing team.
        opponent_team_id : int
            Team ID of the attacking team.

        Returns
        -------
        pd.DataFrame
            Standardised signal output with one row per player-block
            combination and ``signal_value`` set to pressing accuracy
            (fraction correct, in [0, 1]).
        """
        cfg = self.pressing_config

        # Step 1: Compute TTI for all defender-attacker pairs
        self.logger.info("Computing Time-To-Intercept (Bekkers TTI)...")
        tti_df = compute_tti(
            df=match_df,
            config=cfg,
            own_team_id=own_team_id,
            opponent_team_id=opponent_team_id,
        )

        if len(tti_df) == 0:
            self.logger.warning(
                "No valid defender-attacker pairs found — returning empty result."
            )
            return pd.DataFrame(
                columns=[
                    "game_id", "block_id", "phase", "player_id",
                    "team_id_opta", "signal_name", "signal_value",
                    "n_frames",
                ]
            )

        self.logger.info("Computed TTI for %d defender-frame rows.", len(tti_df))

        # Step 2: Detect pressing events
        self.logger.info("Detecting pressing events...")
        pressing_df = detect_pressing_events(
            df=match_df,
            tti_df=tti_df,
            config=cfg,
        )

        n_pressing = pressing_df["is_pressing"].sum()
        self.logger.info("Found %d pressing frames.", n_pressing)

        # Step 3: Classify pressing accuracy
        self.logger.info("Classifying press accuracy...")
        classified_df = classify_pressing_accuracy(
            df=match_df,
            tti_df=tti_df,
            pressing_df=pressing_df,
            config=cfg,
        )

        # Step 4: Aggregate to blocks
        self.logger.info("Aggregating pressing metrics by block...")
        output_df = aggregate_pressing_by_block(
            df=classified_df,
            blocks=blocks,
            config=cfg,
            game_id=game_id,
        )

        self.logger.info(
            "Computed pressing_accuracy signal: %d rows.", len(output_df)
        )
        return output_df

    def validate(self, output_df: pd.DataFrame) -> bool:
        """Validate the pressing accuracy output.

        In addition to the standard schema checks, verifies that:
        - ``signal_value`` is in [0, 1] (fraction of correct presses).
        - ``n_presses`` is non-negative.
        - ``mean_intercept_prob`` (if present) is in [0, 1].

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

        if sv.min() < 0.0 or sv.max() > 1.0:
            raise ValueError(
                f"signal_value must be in [0, 1] (fraction correct), "
                f"got range [{sv.min():.3f}, {sv.max():.3f}]"
            )

        if "n_presses" in output_df.columns:
            if output_df["n_presses"].min() < 0:
                raise ValueError("n_presses contains negative values.")

        if "mean_intercept_prob" in output_df.columns:
            mip = output_df["mean_intercept_prob"]
            if mip.min() < 0.0 or mip.max() > 1.0:
                raise ValueError(
                    f"mean_intercept_prob must be in [0, 1], "
                    f"got range [{mip.min():.3f}, {mip.max():.3f}]"
                )

        return True
