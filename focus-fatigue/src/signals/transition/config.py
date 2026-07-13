"""Configuration parameters for Signal 5 — Transition Recognition.

All tuning parameters are collected here for easy adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransitionConfig:
    """Configuration for transition recognition (Signal 5).

    Controls how possession-change events are detected and how defender
    reaction latency is measured.
    """

    # ── Reaction Detection ──────────────────────────────────────────────

    reaction_window_s: float = 2.0
    """How many seconds of data after a possession flip to look for a
    defender reaction. Default 2.0 s = 50 frames @ 25fps."""

    min_reaction_speed: float = 0.5
    """Minimum speed (m/s) to consider a defender 'actively moving'
    when detecting the onset of their reaction."""

    direction_smoothing_frames: int = 5
    """Number of frames over which to smooth velocity for direction
    estimation when evaluating heading change."""

    reorientation_threshold_deg: float = 45.0
    """Minimum heading change (degrees) from pre-transition heading
    required to consider the defender as having 'recognised' the event."""

    max_reaction_time_s: float = 5.0
    """Maximum accepted reaction time (seconds). Values above this
    threshold are treated as outliers and marked invalid."""

    # ── Frame Rate ─────────────────────────────────────────────────────

    frames_per_second: int = 25
    """Frame rate of the tracking data (Hz)."""

    # ── Surprise Classification ─────────────────────────────────────────

    surprise_ball_speed_threshold: float = 10.0
    """Ball speed (m/s) above which a transition is classified as
    'surprise' (indicating a fast counter-attack)."""

    max_surprise_frames: int = 5
    """Number of frames before a transition to sample ball velocity
    for surprise classification."""

    # ── Min Gap ─────────────────────────────────────────────────────────

    min_gap_frames: int = 10
    """Minimum number of frames between separate transition events.
    Flips closer than this are grouped into a single transition."""


# Default global config instance
DEFAULT_TRANSITION_CONFIG = TransitionConfig()
