"""Data loader for synchronised event data.

Expected format (Hudl event data, synchronised with tracking):
    Provides on-ball events with timestamps aligned to tracking frames.

Returns:
    pd.DataFrame with columns:
        game_id: str       — match identifier
        frame: int         — synchronised tracking frame number
        timestamp: float   — match time in seconds
        event_type: str    — pass, shot, tackle, dribble, etc.
        player_id: str     — player who performed the action
        x: float           — event location x (canonicalised)
        y: float           — event location y (canonicalised)
        outcome: bool      — success/failure where applicable
        period: int        — 1 (first half) or 2 (second half)
"""

from pathlib import Path
from typing import Union

import pandas as pd


def load_events(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load synchronised event data from a single match file.

    Parameters
    ----------
    filepath : str or Path
        Path to the event data file. Expected formats:
        - .json (StatsBomb or Hudl format)
        - .csv
        - .parquet

    Returns
    -------
    pd.DataFrame
        Long-format event data with columns:
        game_id, frame, timestamp, event_type, player_id, x, y, outcome, period

    Notes
    -----
    - Events are synchronised to tracking frames via the frame column.
    - Pitch coordinates use same canonicalisation as tracking [0, 105] x [0, 68].
    - 'outcome' is bool where applicable (True = success, False = failure);
      NaN for neutral events (e.g. fouls, cards).

    Raises
    ------
    NotImplementedError
        Until actual data schema is available.
    """
    raise NotImplementedError(
        "Event loader needs real data schema. "
        "Implement once Hudl data format is known."
    )
