"""Pluggable metric loggers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Protocol

__all__ = ["ConsoleLogger", "JsonlLogger", "Logger", "MultiLogger", "NullLogger"]


class Logger(Protocol):
    """Receives a flat dict of scalar metrics at a given step."""

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        """Record ``metrics`` at ``step``."""
        ...

    def close(self) -> None:
        """Flush and release resources."""
        ...


class NullLogger:
    """Discards everything."""

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        """Do nothing."""
        del step, metrics

    def close(self) -> None:
        """Do nothing."""


class ConsoleLogger:
    """Prints one line per call with selected keys (all keys if ``keys`` is None)."""

    def __init__(self, keys: Sequence[str] | None = None) -> None:
        """Choose which metrics to print."""
        self.keys: Sequence[str] | None = keys

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        """Print ``metrics``."""
        keys = self.keys if self.keys is not None else sorted(metrics)
        parts = [f"{k}={metrics[k]:.4g}" for k in keys if k in metrics]
        print(f"[{step:>10}] " + "  ".join(parts), flush=True)

    def close(self) -> None:
        """Nothing to release."""


class JsonlLogger:
    """Appends one JSON object per call to a file."""

    def __init__(self, path: str | Path) -> None:
        """Open ``path`` for appending (parent directories are created)."""
        self.path: Path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = self.path.open("a", encoding="utf-8")

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        """Write ``{"step": step, **metrics}`` (non-finite floats become null)."""
        row = {"step": step, **{k: (v if math.isfinite(v) else None) for k, v in metrics.items()}}
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the file."""
        self._file.close()


class MultiLogger:
    """Fans out to several loggers."""

    def __init__(self, loggers: Sequence[Logger]) -> None:
        """Wrap ``loggers``."""
        self.loggers: list[Logger] = list(loggers)

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        """Forward to every logger."""
        for logger in self.loggers:
            logger.log(step, metrics)

    def close(self) -> None:
        """Close every logger."""
        for logger in self.loggers:
            logger.close()
