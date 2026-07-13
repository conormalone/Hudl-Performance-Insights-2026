"""Configuration parameters for Model 1 — Pressure Exposure.

All tuning parameters are collected here for easy adjustment.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PressureConfig:
    """Configuration for pressure exposure computation.

    All distances in metres (DOP-normalised pitch coordinates).
    All times in seconds.
    """

    # --- Opponent Proximity ---
    opponent_radius: float = 7.0
    """Radius (m) around a defender to count opposing players."""

    # --- Defensive Depth ---
    goal_line_x: float = -57.0
    """Own goal line x-coordinate (DOP-normalised, all attacks left→right)."""

    # --- Reorientation Detection ---
    reorientation_angle_threshold: float = 45.0
    """Minimum heading change (degrees) to count as a reorientation."""
    reorientation_window_s: float = 1.0
    """Time window (seconds) for detecting sharp heading changes."""
    reorientation_min_speed: float = 0.5
    """Minimum speed (m/s) to consider for reorientation detection."""
    reorientation_frames: int = 25
    """Number of frames in reorientation window (1s @ 25fps)."""

    # --- Transition Detection ---
    transition_zone_radius: float = 7.0
    """Radius (m) around a defender to count a zone transition."""

    # --- Block Segmentation ---
    block_window_minutes: int = 5
    """Duration of each analysis block in minutes."""
    block_min_frames: int = 100
    """Minimum frames required for a valid block."""

    # --- Baselines ---
    baseline_minutes: int = 15
    """Minutes of phase 1 used for per-match baseline."""

    # --- Composite ---
    composite_pressure_formula: str = "1 + sum(indicator / baseline)"
    """Formula for weighted pressure composite."""

    # --- Classification ---
    high_pressure_quantile: float = 0.75
    """Blocks above this quantile are classified as high pressure."""
    low_pressure_quantile: float = 0.25
    """Blocks below this quantile are classified as low pressure (control)."""

    # --- Paths ---
    sample_dir: str = "./data/raw/tracking/sample"
    """Directory containing sample match tracking data."""
    tracking_dir: str = "./data/raw/tracking"
    """Directory containing all match tracking data."""
    output_dir: str = "./outputs/pressure_exposure"
    """Output directory for results."""

    # --- Frame Rate ---
    frame_interval_s: float = 0.04
    """Time between frames (seconds) at 25fps."""
    frames_per_second: int = 25
    """Frame rate of the tracking data."""


# Default global config instance
DEFAULT_CONFIG = PressureConfig()
