"""
IRIS Shared Terminal Logging Utilities
======================================
Headless-safe terminal logging helpers enforcing canonical top-level module
prefixes across IRIS-Viewer, IRIS-Phenotyper, and IRIS-Analyzer.

Format:
    ``[IRIS-Viewer] [Segmentation] Loaded 128 cells.``
    ``[IRIS-Phenotyper] [Gating] [WARN] Missing marker column; skipping.``
"""
from __future__ import annotations

from typing import Dict, Optional, TextIO, Tuple

ALLOWED_MODULES: Tuple[str, ...] = (
    "IRIS-Viewer",
    "IRIS-Phenotyper",
    "IRIS-Analyzer",
)

_MODULE_ALIASES: Dict[str, str] = {
    "iris-viewer": "IRIS-Viewer",
    "iris_viewer": "IRIS-Viewer",
    "viewer": "IRIS-Viewer",
    "iris-phenotyper": "IRIS-Phenotyper",
    "iris_phenotyper": "IRIS-Phenotyper",
    "phenotyper": "IRIS-Phenotyper",
    "iris-analyzer": "IRIS-Analyzer",
    "iris_analyzer": "IRIS-Analyzer",
    "analyzer": "IRIS-Analyzer",
}

_LEVEL_ALIASES: Dict[str, str] = {
    "INFO": "INFO",
    "WARN": "WARN",
    "WARNING": "WARN",
    "ERROR": "ERROR",
    "ERR": "ERROR",
    "DEBUG": "DEBUG",
}


def resolve_iris_module(module: str) -> str:
    """Resolve a module name or short alias to one of ``ALLOWED_MODULES``.

    Raises
    ------
    ValueError
        If *module* does not match one of the three canonical IRIS modules.
    """
    if not isinstance(module, str):
        raise ValueError(
            f"Unknown IRIS module {module!r}; expected one of {ALLOWED_MODULES}."
        )
    cleaned = module.strip()
    if cleaned in ALLOWED_MODULES:
        return cleaned

    normalized = _MODULE_ALIASES.get(cleaned.lower().replace(" ", "-"))
    if normalized is not None:
        return normalized

    raise ValueError(
        f"Unknown IRIS module {module!r}; expected one of {ALLOWED_MODULES}."
    )


def _normalize_submodule(submodule: str) -> str:
    """Normalize a submodule label, stripping whitespace and stray brackets."""
    if not isinstance(submodule, str):
        raise ValueError(f"IRIS log submodule must be a string, got {type(submodule).__name__}.")
    cleaned = submodule.strip().strip("[]").strip()
    if not cleaned:
        raise ValueError("IRIS log submodule must be a non-empty string.")
    return cleaned


def _normalize_level(level: Optional[str]) -> Optional[str]:
    """Normalize an optional log level tag."""
    if level is None:
        return None
    if not isinstance(level, str):
        raise ValueError(f"IRIS log level must be a string or None, got {type(level).__name__}.")
    cleaned = level.strip().strip("[]").strip().upper()
    if not cleaned:
        return None
    return _LEVEL_ALIASES.get(cleaned, cleaned)


def format_iris_log(
    module: str,
    submodule: str,
    message: str,
    *,
    level: Optional[str] = None,
) -> str:
    """Format a standardized IRIS terminal log message."""
    mod_prefix = resolve_iris_module(module)
    sub_prefix = _normalize_submodule(submodule)
    lvl_prefix = _normalize_level(level)
    msg_text = str(message)

    if lvl_prefix is not None:
        return f"[{mod_prefix}] [{sub_prefix}] [{lvl_prefix}] {msg_text}"
    return f"[{mod_prefix}] [{sub_prefix}] {msg_text}"


def iris_log(
    module: str,
    submodule: str,
    message: str,
    *,
    level: Optional[str] = None,
    flush: bool = True,
    file: Optional[TextIO] = None,
) -> str:
    """Print a formatted IRIS log line to terminal and return the formatted string."""
    formatted = format_iris_log(module, submodule, message, level=level)
    if file is None:
        print(formatted, flush=flush)
    else:
        print(formatted, flush=flush, file=file)
    return formatted


def log_info(
    module: str,
    submodule: str,
    message: str,
    *,
    flush: bool = True,
    file: Optional[TextIO] = None,
) -> str:
    """Print an informational IRIS log line (`[Module] [Submodule] message`)."""
    return iris_log(module, submodule, message, level=None, flush=flush, file=file)


def log_warn(
    module: str,
    submodule: str,
    message: str,
    *,
    flush: bool = True,
    file: Optional[TextIO] = None,
) -> str:
    """Print a warning IRIS log line (`[Module] [Submodule] [WARN] message`)."""
    return iris_log(module, submodule, message, level="WARN", flush=flush, file=file)


def log_error(
    module: str,
    submodule: str,
    message: str,
    *,
    flush: bool = True,
    file: Optional[TextIO] = None,
) -> str:
    """Print an error IRIS log line (`[Module] [Submodule] [ERROR] message`)."""
    return iris_log(module, submodule, message, level="ERROR", flush=flush, file=file)


__all__ = [
    "ALLOWED_MODULES",
    "resolve_iris_module",
    "format_iris_log",
    "iris_log",
    "log_info",
    "log_warn",
    "log_error",
]
