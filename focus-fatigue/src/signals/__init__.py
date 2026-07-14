"""Signal framework — base classes, config, output schema, registry, and implementations.

Every signal inherits from SignalBase and is auto-registered via @register_signal.
"""

from .base import SignalBase
from .config import SignalConfig, DEFAULT_SIGNAL_CONFIG
from .output_schema import OUTPUT_COLUMNS, EXPECTED_TYPES, validate_output
from .registry import SIGNAL_REGISTRY, register_signal, get_signal, list_signals

__all__ = [
    "SignalConfig", "DEFAULT_SIGNAL_CONFIG",
    "OUTPUT_COLUMNS", "EXPECTED_TYPES", "validate_output",
    "SIGNAL_REGISTRY", "register_signal", "get_signal", "list_signals",
    "SignalBase",
]
