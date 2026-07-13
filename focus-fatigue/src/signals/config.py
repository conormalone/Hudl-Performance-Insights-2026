"""Configuration parameters for the signal framework.

All tuning parameters are collected here for easy adjustment.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SignalConfig:
    """Configuration for signal computation and block processing.

    Controls how tracking data is segmented into blocks for signal analysis.
    Paths are configurable and default to a standard output layout.
    """

    # --- Block Parameters ---
    block_window_minutes: int = 5
    """Duration of each analysis block in minutes."""

    block_min_frames: int = 100
    """Minimum number of frames required for a valid block."""

    # --- Frame Rate ---
    frames_per_second: int = 25
    """Frame rate of the tracking data."""

    frame_interval_s: float = 0.04
    """Time between frames (seconds) at 25 fps."""

    # --- Paths ---
    output_root: str = "outputs/signals"
    """Root directory for signal outputs (relative or absolute path)."""

    # --- Validation ---
    validate_on_save: bool = True
    """Whether to run validate() automatically before save()."""

    # --- Logging ---
    log_level: str = "INFO"
    """Logging level for signal computation (DEBUG, INFO, WARNING, ERROR)."""


# Default global config instance
DEFAULT_SIGNAL_CONFIG = SignalConfig()
