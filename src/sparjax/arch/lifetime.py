"""Cooperative shutdown and error propagation for actor/learner threads.

Adapted in spirit from Stoix's ``ThreadLifetime`` (Apache-2.0, see ``NOTICE``): a shared
stop flag every thread polls. On top of that, :class:`Lifetime` records the first
exception raised in any thread, so the main thread can re-raise it instead of hanging.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

__all__ = ["Lifetime", "StopRequestedError"]


class StopRequestedError(Exception):
    """Raised inside a blocking wait when shutdown has been requested."""


class Lifetime:
    """Shared stop flag plus first-error capture for a group of threads."""

    def __init__(self) -> None:
        """Create a lifetime in the running state."""
        self._stop: threading.Event = threading.Event()
        self._lock: threading.Lock = threading.Lock()
        self._error: BaseException | None = None
        self._threads: list[threading.Thread] = []

    @property
    def stopped(self) -> bool:
        """Whether shutdown has been requested."""
        return self._stop.is_set()

    @property
    def error(self) -> BaseException | None:
        """The first exception raised by a supervised thread, if any."""
        return self._error

    def stop(self) -> None:
        """Request shutdown of every thread sharing this lifetime."""
        self._stop.set()

    def wait(self, timeout: float) -> bool:
        """Sleep up to ``timeout`` seconds, waking early on stop. Returns ``stopped``."""
        return self._stop.wait(timeout)

    def check(self) -> None:
        """Raise :class:`StopRequestedError` if shutdown has been requested."""
        if self._stop.is_set():
            raise StopRequestedError

    def fail(self, exc: BaseException) -> None:
        """Record ``exc`` (first one wins) and request shutdown."""
        with self._lock:
            if self._error is None:
                self._error = exc
        self.stop()

    def spawn(self, target: Callable[..., Any], name: str, *args: Any) -> threading.Thread:
        """Start a daemon thread whose exceptions are captured by :meth:`fail`."""

        def run() -> None:
            try:
                target(*args)
            except StopRequestedError:
                pass
            except BaseException as exc:  # propagated to the main thread via self.error
                self.fail(exc)

        thread = threading.Thread(target=run, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()
        return thread

    def join(self, timeout: float = 10.0) -> list[str]:
        """Stop and join all spawned threads. Returns names of threads still alive."""
        self.stop()
        for thread in self._threads:
            thread.join(timeout)
        return [t.name for t in self._threads if t.is_alive()]

    def raise_if_failed(self) -> None:
        """Re-raise the first captured thread exception in the calling thread."""
        if self._error is not None:
            raise RuntimeError(f"a worker thread failed: {self._error!r}") from self._error
