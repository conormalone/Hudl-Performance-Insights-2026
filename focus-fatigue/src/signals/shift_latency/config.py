"""Configuration parameters for Signal 2 — Shift Latency.

All tuning parameters are collected here for easy adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ShiftLatencyConfig:
    """Configuration for shift latency (Signal 2).

    Controls ball speed spike detection, opponent run detection, and
    defender reaction latency measurement.
    """

    # ── Ball Speed Spike Detection ────────────────────────────────────

    ball_speed_spike_threshold: float = 15.0
    """Ball speed (m/s) above which a frame is considered a 'spike'.
    Typical pass speed: 10-25 m/s. Professional long passes can reach
    30 m/s. Default 15 m/s filters out casual passes."""

    ball_speed_smoothing_frames: int = 5
    """Number of frames for rolling average of ball speed, to smooth
    out single-frame measurement noise."""

    min_spike_gap_frames: int = 25
    """Minimum gap (frames = 1 second at 25fps) between separate spike
    events. Nearby spikes are grouped into a single event."""

    # ── Opponent Run Detection ────────────────────────────────────────

    opponent_run_speed_threshold: float = 5.0
    """Speed (m/s) above which an opponent is considered to be making
    an 'aggressive run' that requires a defensive reaction."""

    opponent_run_acceleration_window: int = 10
    """Number of frames over which to check acceleration for determining
    run intensity."""

    # ── Reaction Detection ────────────────────────────────────────────

    reaction_window_s: float = 3.0
    """How many seconds after a spike/run to look for defender reaction.
    Longer than Signal 5 (2s) because shift reactions can be slower
    (no clear possession flip as cue). Default 3.0 s = 75 frames."""

    min_reaction_speed: float = 0.5
    """Minimum speed (m/s) for a defender to be considered 'reacting'."""

    direction_smoothing_frames: int = 5
    """Frames for smoothing direction estimation."""

    reorientation_threshold_deg: float = 30.0
    """Minimum heading change (degrees) for reaction detection.
    Slightly lower than Signal 5's 45° because shifts are subtler."""

    max_reaction_time_s: float = 5.0
    """Maximum plausible reaction time (seconds)."""

    # ── Frame Rate ─────────────────────────────────────────────────────

    frames_per_second: int = 25
    """Frame rate of the tracking data (Hz)."""


# Default global config
DEFAULT_SHIFT_LATENCY_CONFIG = ShiftLatencyConfig()
