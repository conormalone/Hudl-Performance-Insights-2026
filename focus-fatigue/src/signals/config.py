"""Configuration parameters for the signal framework.

All tuning parameters are collected here for easy adjustment.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SignalConfig:
    """Configuration for signal computation and block processing.

    Block/frame parameters are inherited from ``PressureConfig`` in
    ``src.pressure.config`` and should be imported from there.
    This config only holds signal-specific settings.
    """

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
