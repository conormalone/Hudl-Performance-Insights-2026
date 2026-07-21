"""Signal registry for the src/signals/ package.

Provides a @register_signal decorator that auto-registers signal
classes into SIGNAL_REGISTRY. Signals from this package produce
stacked-format output:

  game_id, block_id, phase, player_id, team_id_opta,
  signal_name, signal_value, n_frames

Usage:
    from src.signals.registry import register_signal

    @register_signal
    class MySignal:
        signal_name = "my_signal"
        ...
"""

import logging
from typing import Type

logger = logging.getLogger(__name__)

# Central registry: signal_name -> signal class
SIGNAL_REGISTRY: dict[str, Type] = {}


def register_signal(cls):
    """Decorator that registers a signal class in SIGNAL_REGISTRY.

    The class must define a class-level `signal_name` attribute.
    Registration is idempotent — re-registering the same name
    overwrites silently.
    """
    name = getattr(cls, "signal_name", None)
    if not name:
        raise ValueError(
            f"@{register_signal.__name__} decorated class {cls.__name__} "
            f"must define a 'signal_name' attribute."
        )
    SIGNAL_REGISTRY[name] = cls
    logger.debug("Registered new signal: '%s' -> %s", name, cls.__name__)
    return cls


def discover_signals() -> dict[str, Type]:
    """Return the populated SIGNAL_REGISTRY.

    Import all signal modules to trigger decorator execution before
    calling this function.
    """
    return dict(SIGNAL_REGISTRY)


def get_signal_names() -> list[str]:
    """Return names of all registered signals."""
    return list(SIGNAL_REGISTRY.keys())


def get_signal(name: str) -> Type:
    """Get a signal class by name. Raises KeyError if not found."""
    if name not in SIGNAL_REGISTRY:
        raise KeyError(
            f"Unknown signal '{name}'. "
            f"Available: {list(SIGNAL_REGISTRY.keys())}"
        )
    return SIGNAL_REGISTRY[name]
