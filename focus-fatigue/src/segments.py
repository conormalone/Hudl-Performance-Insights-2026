"""Match segmentation into analysis blocks.

Divides a match into fixed-width time blocks (default 5 minutes)
for per-block signal computation.
"""

from typing import Optional

import numpy as np
import pandas as pd

# Each frame is 0.04s (25fps)
FRAME_INTERVAL_S = 0.04
FRAMES_PER_MIN = int(60 / FRAME_INTERVAL_S)  # 1500


def split_into_blocks(
    df: pd.DataFrame,
    window_minutes: int = 5,
    min_frames: int = 100,
    phase_col: str = "phase",
    time_col: str = "time",
    frame_col: str = "frame_count",
) -> list[pd.DataFrame]:
    """Split a match DataFrame into fixed-width time blocks.

    Parameters
    ----------
    df : pd.DataFrame
        Tracking data for a single match.
    window_minutes : int, default 5
        Duration of each analysis block in minutes.
    min_frames : int, default 100
        Minimum frames required for a block to be valid.
    phase_col : str, default 'phase'
        Column name for match phase (1 or 2).
    time_col : str, default 'time'
        Column name for time in seconds.
    frame_col : str, default 'frame_count'
        Column name for frame counter.

    Returns
    -------
    list of pd.DataFrame
        Each element is a DataFrame for one block, with an added
        'block_id' column: f"{phase}_{block_num}".
    """
    blocks = []
    frame_interval = window_minutes * 60 / FRAME_INTERVAL_S

    for phase in sorted(df[phase_col].unique()):
        phase_df = df[df[phase_col] == phase].copy()
        phase_frames = phase_df[frame_col].unique()
        phase_frames.sort()

        if len(phase_frames) == 0:
            continue

        frame_min = phase_frames.min()
        frame_max = phase_frames.max()

        block_start = frame_min
        block_num = 0

        while block_start <= frame_max:
            block_end = block_start + frame_interval
            block_df = phase_df[
                (phase_df[frame_col] >= block_start)
                & (phase_df[frame_col] < block_end)
            ].copy()

            if len(block_df) >= min_frames:
                block_df["block_id"] = f"{phase}_{block_num}"
                blocks.append(block_df)

            block_start = block_end
            block_num += 1

    return blocks


def block_summary(blocks: list[pd.DataFrame]) -> pd.DataFrame:
    """Create a summary table of all blocks.

    Returns a DataFrame with one row per block:
    block_id, phase, block_num, n_frames, time_start, time_end.
    """
    records = []
    for blk in blocks:
        bid = blk["block_id"].iloc[0]
        phase, num = bid.split("_")
        records.append(
            {
                "block_id": bid,
                "phase": int(phase),
                "block_num": int(num),
                "n_frames": len(blk),
                "time_start": blk["time"].min(),
                "time_end": blk["time"].max(),
            }
        )
    return pd.DataFrame(records)
