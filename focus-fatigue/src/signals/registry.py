"""Signal registry — maps human-readable signal names to their classes.

Example
-------
Usage::

    from src.signals.registry import register_signal, get_signal, list_signals

    @register_signal
    class TransitionLatency(SignalBase):
        signal_name = "transition_latency"
        ...

    cls = get_signal("transition_latency")
    assert cls is TransitionLatency
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import SignalBase

# ── Registry Store ──────────────────────────────────────────────────────────

SIGNAL_REGISTRY: dict[str, type["SignalBase"]] = {}
"""Global registry mapping ``signal_name`` → ``SignalBase`` subclass."""


# ── Decorator / Registration ───────────────────────────────────────────────

def register_signal(
    cls: type["SignalBase"],
) -> type["SignalBase"]:
    """Register a :class:`SignalBase` subclass in the global registry.

    Can be used as a decorator::

        @register_signal
        class MySignal(SignalBase):
            signal_name = "my_signal"
            ...

    Parameters
    ----------
    cls : type[SignalBase]
        A concrete subclass of :class:`SignalBase` with a non-empty
        ``signal_name`` class attribute.

    Returns
    -------
    type[SignalBase]
        The same class, unchanged (for decorator convenience).

    Raises
    ------
    ValueError
        If the class has an empty ``signal_name``, or if a different class
        is already registered under the same name.
    """
    name = cls.signal_name
    if not name:
        raise ValueError(
            f"Cannot register {cls.__name__}: signal_name is empty."
        )

    existing = SIGNAL_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Cannot register {cls.__name__} as '{name}' — "
            f"already registered to {existing.__module__}.{existing.__qualname__}"
        )

    SIGNAL_REGISTRY[name] = cls
    return cls


# ── Lookup ──────────────────────────────────────────────────────────────────


def get_signal(name: str) -> type["SignalBase"]:
    """Retrieve a signal class by its name.

    Parameters
    ----------
    name : str
        The signal name (must match the ``signal_name`` class attribute).

    Returns
    -------
    type[SignalBase]
        The registered signal class.

    Raises
    ------
    KeyError
        If no signal is registered under the given name.
    """
    if name not in SIGNAL_REGISTRY:
        available = ", ".join(sorted(SIGNAL_REGISTRY))
        raise KeyError(
            f"Unknown signal '{name}'. "
            f"Registered signals: [{available}]"
        )
    return SIGNAL_REGISTRY[name]


def list_signals() -> list[str]:
    """Return a sorted list of all registered signal names.

    Returns
    -------
    list[str]
        Sorted signal names.
    """
    return sorted(SIGNAL_REGISTRY.keys())
