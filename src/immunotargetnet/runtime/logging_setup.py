"""Logging configuration for library and CLI entry points."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(level: str = "INFO", *, stream: bool = True) -> None:
    root = logging.getLogger("immunotargetnet")
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
    if stream:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"immunotargetnet.{name}" if name else "immunotargetnet")
