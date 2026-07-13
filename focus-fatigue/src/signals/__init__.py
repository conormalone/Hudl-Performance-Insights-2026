"""Signal framework — base classes, config, output schema, and registry.

All signal implementations live under ``src/signals/`` and inherit from
:class:`SignalBase`. The registry enables dynamic discovery and loading
of signals by name.
"""

from .base import SignalBase
from .config import SignalConfig, DEFAULT_SIGNAL_CONFIG
from .output_schema import (
    OUTPUT_COLUMNS,
    EXPECTED_TYPES,
    OUTPUT_COLUMNS as OUTPUT_SCHEMA_COLUMNS,  # alias for readability
    validate_output,
)
from .registry import SIGNAL_REGISTRY, register_signal, get_signal, list_signals

__all__ = [
    # Config
    "SignalConfig",
    "DEFAULT_SIGNAL_CONFIG",
    # Output schema
    "OUTPUT_COLUMNS",
    "OUTPUT_SCHEMA_COLUMNS",
    "EXPECTED_TYPES",
    "validate_output",
    # Registry
    "SIGNAL_REGISTRY",
    "register_signal",
    "get_signal",
    "list_signals",
    # Base
    "SignalBase",
]
