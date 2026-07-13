"""Configuration parameters for Signal 1 — Positional Drift.

Controls how expected positions from shape.json are mapped to tracking
frames and how drift distance is computed.

The shape files provide pre-computed ``averageRolePositionX/Y`` values
for each player's expected position at ~1-minute intervals. The drift
signal measures how far a player's actual position is from their
role-based expected position when their team is out of possession.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DriftConfig:
    """Configuration for positional drift computation.

    Controls shape entry windowing, fit-quality filtering, and frame-rate
    assumptions for the bridge between tracking data and shape entries.
    """

    # ── Shape Window ───────────────────────────────────────────────────

    shape_window_s: float = 60.0
    """How many seconds a single shape entry covers. Shape entries are
    produced approximately once per minute. Default 60.0 seconds."""

    # ── Fit Score Threshold ────────────────────────────────────────────

    min_fit_score: float = 0.5
    """Minimum fit score (0–1) to trust a shape entry. Entries below
    this threshold are treated as unreliable and return NaN for
    expected positions. Lower values mean the player's actual position
    was poorly explained by the role centroid at that time."""

    # ── Frame Rate ─────────────────────────────────────────────────────

    frames_per_second: int = 25
    """Frame rate of the tracking data (Hz). Used to convert frame
    numbers to time-based minute windows for shape lookups."""

    @property
    def frames_per_minute(self) -> int:
        """Number of tracking frames per minute at the configured FPS."""
        return self.frames_per_second * 60

    # ── Valid Range ────────────────────────────────────────────────────

    max_plausible_drift_m: float = 50.0
    """Maximum plausible drift distance in metres. Values above this
    are likely artifacts (e.g. swapped player IDs, incorrect shape
    match) and are clipped to NaN."""


# Default global config instance
DEFAULT_DRIFT_CONFIG = DriftConfig()
