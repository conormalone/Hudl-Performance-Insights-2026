"""Signal 1 — Positional Drift.

Measures how far defenders drift from their expected (shape-model)
positions during out-of-possession phases. The core hypothesis is
that cognitively fatigued defenders exhibit degraded spatial awareness,
manifesting as larger positional errors relative to the ideal role
centroid provided by the shape.json model.

Unlike Signals 2+ (which require EFPI clustering and Hungarian
assignment), this signal uses the pre-computed ``averageRolePositionX/Y``
values directly from the shape.json files — giving us the expected
position at ~1-minute resolution without any unsupervised learning.

The pipeline runs:

1. :func:`bridge.load_shape_file` — Parse the shape.json file.
2. :func:`bridge.build_player_role_map` — Bridge tracking players to
   shape roles via team UUID + jersey number.
3. :func:`drift.compute_drift` — Compute per-frame Euclidean drift.
4. :func:`drift.aggregate_drift_by_block` — Summarise into per-block,
   per-player statistics.

See Also
--------
:mod:`bridge` — Player-to-shape-role bridge logic.
:mod:`drift` — Drift computation and aggregation.
:mod:`config` — Configuration dataclass.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..base import SignalBase
from ..config import SignalConfig, DEFAULT_SIGNAL_CONFIG
from ..registry import register_signal
from .config import DriftConfig, DEFAULT_DRIFT_CONFIG
from .bridge import load_shape_file, build_player_role_map
from .drift import compute_drift, aggregate_drift_by_block

# ── Public API ──────────────────────────────────────────────────────────────

__all__ = [
    # Config
    "DriftConfig",
    "DEFAULT_DRIFT_CONFIG",
    # Bridge
    "load_shape_file",
    "build_player_role_map",
    # Drift
    "compute_drift",
    "aggregate_drift_by_block",
    # Class
    "PositionalDriftSignal",
]


# ═══════════════════════════════════════════════════════════════════════════
# Registered Signal Class
# ═══════════════════════════════════════════════════════════════════════════


@register_signal
class PositionalDriftSignal(SignalBase):
    """Signal 1 — Positional Drift (spatial awareness degradation).

    Computes per-defender positional drift across match blocks using
    shape-model expected positions. ``signal_value`` is set to the
    mean Euclidean drift distance (metres) for a player-block
    combination when out of possession.

    A larger drift indicates that the defender is further from their
    expected role position — a potential sign of degraded spatial
    awareness (cognitive fatigue).

    Parameters
    ----------
    signal_config : SignalConfig | None
        Base signal framework configuration.
    drift_config : DriftConfig | None
        Drift-specific configuration (FPS, fit thresholds, etc.).
    logger : logging.Logger | None
        Logger instance.
    """

    signal_name = "positional_drift"
    """Registered signal name used in the output schema and registry."""

    def __init__(
        self,
        signal_config: SignalConfig | None = None,
        drift_config: DriftConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(config=signal_config, logger=logger)
        self.drift_config = drift_config or DEFAULT_DRIFT_CONFIG

    # ── Public API ─────────────────────────────────────────────────────────

    def compute(
        self,
        match_df: pd.DataFrame,
        blocks: list[dict[str, Any]],
        *,
        game_id: str = "",
        shape_path: str = "",
        player_role_map: dict[int, dict[int, dict[str, Any]]] | None = None,
        team_id_col: str = "team_id_opta",
        team_in_possession_col: str = "team_in_possession",
    ) -> pd.DataFrame:
        """Compute positional drift across all blocks.

        Parameters
        ----------
        match_df : pd.DataFrame
            Full tracking data for a single match. Must contain:
            ``frame``, ``player_id``, ``team_id_opta`` (or
            ``team_id_col``), ``x``, ``y``, and DOP-normalised
            possession indicator.
        blocks : list[dict[str, Any]]
            Block definitions (see :class:`SignalBase.compute`).
        game_id : str, optional
            Match identifier for the output.
        shape_path : str
            Path to the shape.json file for this match. Required if
            ``player_role_map`` is not provided.
        player_role_map : dict, optional
            Pre-built player role map (from
            :func:`bridge.build_player_role_map`). If not provided,
            the signal builds it from ``shape_path`` and ``match_df``.
        team_id_col : str
            Column name for team Opta ID.
        team_in_possession_col : str
            Column name for DOP-normalised team possession indicator.

        Returns
        -------
        pd.DataFrame
            Standardised signal output with one row per player-block
            combination and ``signal_value`` set to mean drift (metres).
        """
        cfg = self.drift_config

        # ── Step 1: Build player role map (if not provided) ──────────────
        if player_role_map is None:
            if not shape_path:
                raise ValueError(
                    "Either 'shape_path' or 'player_role_map' must be provided."
                )

            self.logger.info("Loading shape file: %s", shape_path)
            shapes = load_shape_file(shape_path)

            self.logger.info("Building player-to-role bridge...")
            player_role_map = build_player_role_map(
                tracking_df=match_df,
                shapes=shapes,
                min_fit_score=cfg.min_fit_score,
            )

            n_players = len(player_role_map)
            n_roles = sum(len(v) for v in player_role_map.values())
            self.logger.info(
                "Bridged %d players with %d role-minute entries.",
                n_players,
                n_roles,
            )

        if not player_role_map:
            self.logger.warning(
                "Player role map is empty — no shape data matched. "
                "Returning empty result."
            )
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
                ]
            )

        # ── Step 2: Compute per-frame drift ─────────────────────────────
        self.logger.info("Computing per-frame positional drift...")
        drift_df = compute_drift(
            df=match_df,
            player_role_map=player_role_map,
            config=cfg,
            team_in_possession_col=team_in_possession_col,
            team_id_col=team_id_col,
        )

        n_valid_drift = drift_df["drift_m"].notna().sum()
        self.logger.info("Computed drift for %d frame-player rows.", n_valid_drift)

        # ── Step 3: Aggregate to blocks ─────────────────────────────────
        self.logger.info("Aggregating drift by block...")
        output_df = aggregate_drift_by_block(
            df=drift_df,
            blocks=blocks,
            config=cfg,
            game_id=game_id,
        )

        self.logger.info(
            "Computed positional_drift signal: %d rows.", len(output_df)
        )
        return output_df

    def validate(self, output_df: pd.DataFrame) -> bool:
        """Validate the positional drift output.

        In addition to the standard schema checks, verifies that:
        - ``signal_value`` (mean drift) is non-negative (distance).
        - Values are within a plausible range (< max_plausible_drift_m).
        - ``drift_p90`` >= ``signal_value`` (statistical sanity).
        - ``mean_fit_score`` is in [0, 1] when present.

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
                f"signal_value (mean drift) contains negative values "
                f"(min={sv.min():.3f}). Drift distance cannot be negative."
            )

        max_allowed = self.drift_config.max_plausible_drift_m
        if sv.max() > max_allowed:
            raise ValueError(
                f"signal_value exceeds max_plausible_drift_m={max_allowed:.1f} m: "
                f"max observed = {sv.max():.3f} m."
            )

        # Extra column validation (non-blocking warnings)
        if "drift_p90" in output_df.columns:
            invalid_p90 = output_df["drift_p90"].notna() & (
                output_df["drift_p90"] < output_df["signal_value"]
            )
            if invalid_p90.any():
                n_bad = invalid_p90.sum()
                self.logger.warning(
                    "%d row(s) have drift_p90 < signal_value (mean). "
                    "This is unusual and may indicate data issues.",
                    n_bad,
                )

        if "mean_fit_score" in output_df.columns:
            mfs = output_df["mean_fit_score"].dropna()
            if len(mfs) > 0 and (mfs.min() < 0.0 or mfs.max() > 1.0):
                raise ValueError(
                    f"mean_fit_score must be in [0, 1], "
                    f"got range [{mfs.min():.3f}, {mfs.max():.3f}]"
                )

        return True

    def compute_pipeline(
        self,
        match_df: pd.DataFrame,
        blocks: list[dict[str, Any]],
        *,
        game_id: str = "",
        shape_path: str = "",
        team_id_col: str = "team_id_opta",
        team_in_possession_col: str = "team_in_possession",
        save: bool = True,
    ) -> pd.DataFrame:
        """Convenience: run the full compute → validate → save pipeline.

        Parameters
        ----------
        match_df : pd.DataFrame
            Full tracking data for a single match.
        blocks : list[dict[str, Any]]
            Block definitions.
        game_id : str
            Match identifier for the output and filename.
        shape_path : str
            Path to the shape.json file for this match.
        team_id_col : str
            Column name for team Opta ID.
        team_in_possession_col : str
            Column name for DOP-normalised possession indicator.
        save : bool
            Whether to save results to disk (default ``True``).

        Returns
        -------
        pd.DataFrame
            The computed output DataFrame.
        """
        self.logger.info(
            "Running positional_drift pipeline: game=%s, shape=%s",
            game_id or "unknown",
            shape_path or "unknown",
        )

        # 1. Compute
        output_df = self.compute(
            match_df=match_df,
            blocks=blocks,
            game_id=game_id,
            shape_path=shape_path,
            team_id_col=team_id_col,
            team_in_possession_col=team_in_possession_col,
        )

        # 2. Validate
        self.validate(output_df)

        # 3. Save
        if save:
            self.save(output_df, match_id=game_id)

        return output_df
