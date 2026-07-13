"""Standardised output schema for all signals.

Every signal computes per-block, per-player values and writes them in
this uniform format for downstream consumption (e.g. aggregation, visualisation).
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

# ── Column Definitions ──────────────────────────────────────────────────────

OUTPUT_COLUMNS: list[str] = [
    "game_id",
    "block_id",
    "phase",
    "player_id",
    "team_id_opta",
    "signal_name",
    "signal_value",
    "n_frames",
]

EXPECTED_TYPES: dict[str, str] = {
    "game_id": "str",
    "block_id": "str",
    "phase": "int",
    "player_id": "int",
    "team_id_opta": "int",
    "signal_name": "str",
    "signal_value": "float",
    "n_frames": "int",
}

# ── Column Metadata ─────────────────────────────────────────────────────────

COLUMN_DESCRIPTIONS: dict[str, str] = {
    "game_id": "Unique match identifier (e.g. match file basename).",
    "block_id": "Block identifier, typically 'block_{n}' where n is the block index.",
    "phase": "Match phase (1 = first half baseline, 2 = second half, etc.).",
    "player_id": "Player identifier from the tracking data.",
    "team_id_opta": "Opta team identifier.",
    "signal_name": "Name of the computed signal (e.g. 'transition_latency').",
    "signal_value": "Float value of the signal for this player-block combination.",
    "n_frames": "Number of tracking frames in this block (used for weighting/sanity).",
}

# ── Validation ──────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: list[str] = sorted(OUTPUT_COLUMNS)
SIGNAL_VALUE_RANGE: tuple[float, float] = (-1e10, 1e10)
"""Sensible float bounds. Individual signals may have narrower valid ranges."""


def validate_output(df: pd.DataFrame, signal_name: str | None = None) -> bool:
    """Validate that a DataFrame conforms to the standard output schema.

    Checks for:
    - All required columns present (supersets are fine; extraneous columns allowed)
    - Correct data types per ``EXPECTED_TYPES``
    - No NaN values in required columns
    - Signal value within the configured valid range

    Parameters
    ----------
    df : pd.DataFrame
        Output DataFrame to validate.
    signal_name : str | None
        If provided, also checks that ``signal_name`` column matches this value.

    Returns
    -------
    bool
        True if the DataFrame passes all validation checks.

    Raises
    ------
    ValueError
        Describing the first validation failure encountered.
    """
    # ── Required columns ────────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"DataFrame has columns: {list(df.columns)}"
        )

    # ── Data types ──────────────────────────────────────────────────────
    type_map = {
        "int": "int64",
        "float": "float64",
        "str": "object",
    }

    for col, expected_type_name in EXPECTED_TYPES.items():
        expected_dtype = type_map[expected_type_name]
        actual_dtype = str(df[col].dtype)

        # Allow subtype compatibility (e.g. int32 vs int64)
        if expected_type_name == "int":
            if not actual_dtype.startswith("int") and not actual_dtype.startswith("Int"):
                raise ValueError(
                    f"Column '{col}' has dtype '{actual_dtype}', "
                    f"expected integer (got {expected_dtype})"
                )
        elif expected_type_name == "float":
            if not actual_dtype.startswith("float") and not actual_dtype.startswith("Float"):
                warnings.warn(
                    f"Column '{col}' has dtype '{actual_dtype}', "
                    f"expected {expected_dtype} — will attempt coercion"
                )
        elif expected_type_name == "str":
            if actual_dtype != "object" and not actual_dtype.startswith("str"):
                raise ValueError(
                    f"Column '{col}' has dtype '{actual_dtype}', "
                    f"expected string/object"
                )

    # ── Missing values ──────────────────────────────────────────────────
    null_counts = df[OUTPUT_COLUMNS].isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        raise ValueError(
            f"Missing values found in required columns:\n{null_cols}"
        )

    # ── Signal value range ──────────────────────────────────────────────
    low, high = SIGNAL_VALUE_RANGE
    out_of_range = ~df["signal_value"].between(low, high)
    if out_of_range.any():
        n = out_of_range.sum()
        raise ValueError(
            f"{n} signal_value(s) outside valid range [{low}, {high}]. "
            f"Min: {df['signal_value'].min()}, Max: {df['signal_value'].max()}"
        )

    # ── signal_name consistency ──────────────────────────────────────────
    if signal_name is not None:
        unique_names = df["signal_name"].unique()
        if len(unique_names) != 1 or unique_names[0] != signal_name:
            raise ValueError(
                f"Expected all signal_name values to be '{signal_name}', "
                f"got: {list(unique_names)}"
            )

    return True
