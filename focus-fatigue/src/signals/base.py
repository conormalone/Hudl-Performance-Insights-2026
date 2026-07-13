"""Abstract base class for all signal implementations.

Every signal (transition latency, pressing efficiency, positional drift, etc.)
inherits from :class:`SignalBase` and implements the ``compute()`` and
``validate()`` methods. The framework handles standardised output serialisation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import SignalConfig, DEFAULT_SIGNAL_CONFIG
from .output_schema import OUTPUT_COLUMNS, validate_output


class SignalBase(ABC):
    """Abstract base class for computing a per-block, per-player signal.

    Subclasses must set :attr:`signal_name` as a class-level attribute and
    implement both :meth:`compute` and :meth:`validate`.

    Example
    -------
    .. code-block:: python

        class TransitionLatency(SignalBase):
            signal_name = "transition_latency"

            def compute(self, match_df, blocks):
                ...
                return output_df

            def validate(self, output_df):
                return super().validate(output_df)
    """

    signal_name: str = ""
    """Unique identifier for this signal (e.g. ``"transition_latency"``).

    Used as the column value in ``signal_name`` in output DataFrames and
    as the sub-directory name when saving results.
    """

    def __init__(
        self,
        config: SignalConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialise the signal with a configuration and optional logger.

        Parameters
        ----------
        config : SignalConfig | None
            Configuration instance. Falls back to ``DEFAULT_SIGNAL_CONFIG``.
        logger : logging.Logger | None
            Logger for status messages. If ``None``, a module-level logger
            named after the subclass is created.
        """
        if not self.signal_name:
            raise ValueError(
                f"{type(self).__name__} must define a non-empty class attribute "
                f"'signal_name'"
            )

        self.config = config or DEFAULT_SIGNAL_CONFIG
        self.logger = logger or logging.getLogger(
            f"{__name__}.{type(self).__name__}"
        )
        self._output_dir: Path | None = None

    # ── Abstract Methods ────────────────────────────────────────────────

    @abstractmethod
    def compute(
        self,
        match_df: pd.DataFrame,
        blocks: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Compute the signal on a per-block, per-player basis.

        Parameters
        ----------
        match_df : pd.DataFrame
            Full tracking data for a single match. Must contain at least
            ``frame``, ``player_id``, ``team_id_opta`` columns and any
            spatiotemporal columns needed by the signal implementation.
        blocks : list[dict[str, Any]]
            List of block definitions. Each block is a dict with at least:
                - ``block_id`` : str  — e.g. ``"block_0"``
                - ``phase`` : int     — match phase (1, 2, ...)
                - ``start_frame`` : int
                - ``end_frame`` : int

        Returns
        -------
        pd.DataFrame
            A DataFrame conforming to the standard output schema
            defined in :mod:`src.signals.output_schema`.
        """
        ...

    @abstractmethod
    def validate(self, output_df: pd.DataFrame) -> bool:
        """Validate the computed output DataFrame.

        The default implementation delegates to
        :func:`src.signals.output_schema.validate_output`.
        Subclasses may override to add signal-specific checks (e.g. value
        ranges, minimum counts).

        Parameters
        ----------
        output_df : pd.DataFrame
            The output DataFrame to validate.

        Returns
        -------
        bool
            ``True`` if the DataFrame passes all validation checks.

        Raises
        ------
        ValueError
            Describing the first validation failure encountered.
        """
        return validate_output(output_df, signal_name=self.signal_name)

    # ── Concrete Methods ────────────────────────────────────────────────

    def save(
        self,
        output_df: pd.DataFrame,
        path: str | Path | None = None,
        *,
        match_id: str | None = None,
    ) -> Path:
        """Save the output DataFrame to a CSV file.

        If ``path`` is provided, writes directly there.
        Otherwise, constructs the path as::

            {output_root}/{signal_name}/{match_id}.csv

        where ``output_root`` comes from ``self.config.output_root``.

        Parameters
        ----------
        output_df : pd.DataFrame
            DataFrame to save (must conform to the output schema).
        path : str | Path | None
            Explicit save path. If ``None``, constructs one from match_id.
        match_id : str | None
            Match identifier used for the filename. Required when ``path``
            is not provided.

        Returns
        -------
        Path
            The resolved path the file was written to.

        Raises
        ------
        ValueError
            If neither ``path`` nor ``match_id`` is provided.
        """
        # Resolve target path
        if path is not None:
            target = Path(path)
        elif match_id is not None:
            root = Path(self.config.output_root)
            target = root / self.signal_name / f"{match_id}.csv"
        else:
            raise ValueError(
                "Either 'path' or 'match_id' must be provided to save()."
            )

        # Ensure parent directory exists
        target.parent.mkdir(parents=True, exist_ok=True)

        # Validate before saving if configured
        if self.config.validate_on_save:
            try:
                self.validate(output_df)
                self.logger.info("Validation passed for %s", target)
            except ValueError as exc:
                self.logger.warning(
                    "Saving %s despite validation failure: %s", target, exc
                )

        # Write CSV
        output_df.to_csv(target, index=False)
        self.logger.info("Saved %d rows to %s", len(output_df), target)
        return target

    def ensure_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has all standard output columns, adding any
        that are missing with sensible defaults.

        Parameters
        ----------
        df : pd.DataFrame
            Partially populated output DataFrame (at minimum must contain
            ``signal_value``).

        Returns
        -------
        pd.DataFrame
            DataFrame guaranteed to have all ``OUTPUT_COLUMNS``.
        """
        defaults: dict[str, Any] = {
            "game_id": "",
            "block_id": "",
            "phase": 0,
            "player_id": 0,
            "team_id_opta": 0,
            "signal_name": self.signal_name,
            "signal_value": np.nan,
            "n_frames": 0,
        }

        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = defaults[col]

        return df[OUTPUT_COLUMNS]

    def run_pipeline(
        self,
        match_df: pd.DataFrame,
        blocks: list[dict[str, Any]],
        *,
        save: bool = True,
        match_id: str | None = None,
        path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Convenience method that runs the full compute→validate→save pipeline.

        Parameters
        ----------
        match_df : pd.DataFrame
            Full tracking data for a single match.
        blocks : list[dict[str, Any]]
            List of block definitions (see :meth:`compute`).
        save : bool
            Whether to save results to disk (default ``True``).
        match_id : str | None
            Match identifier used for the filename (required if ``save=True``
            and ``path`` is not provided).
        path : str | Path | None
            Explicit save path (overrides automatic path construction).

        Returns
        -------
        pd.DataFrame
            The computed output DataFrame.
        """
        self.logger.info(
            "Running pipeline: signal=%s, match=%s",
            self.signal_name,
            match_id or "unknown",
        )

        # 1. Compute
        output_df = self.compute(match_df, blocks)

        # 2. Validate
        self.validate(output_df)
        self.logger.info(
            "Computed %d rows for signal '%s'", len(output_df), self.signal_name
        )

        # 3. Save
        if save:
            self.save(output_df, path=path, match_id=match_id)

        return output_df
