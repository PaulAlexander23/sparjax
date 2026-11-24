"""Run evaluation in a background thread so it never stalls the learner.

The learner submits parameter snapshots with :meth:`~AsyncEvaluator.submit`. The evaluator keeps
only the newest pending snapshot: if evaluation is slower than submission, intermediate
snapshots are skipped rather than queued. :meth:`~AsyncEvaluator.finish` evaluates the
last submitted snapshot before returning, so the final parameters are always evaluated.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from sparjax.arch.lifetime import Lifetime

__all__ = ["AsyncEvaluator"]

EvalFn = Callable[[int, Any], dict[str, float]]
Sink = Callable[[int, int, dict[str, float]], None]


class AsyncEvaluator:
    """A background thread that evaluates the newest submitted parameters.

    Args:
        eval_fn: ``(update, params) -> metrics``.
        sink: Receives ``(update, step, metrics)`` for every completed evaluation.
        lifetime: Shared lifetime; evaluation errors are reported through it.
    """

    def __init__(self, eval_fn: EvalFn, sink: Sink, lifetime: Lifetime) -> None:
        """Start the evaluator thread."""
        self._eval_fn: EvalFn = eval_fn
        self._sink: Sink = sink
        self._lifetime: Lifetime = lifetime
        self._cond: threading.Condition = threading.Condition()
        self._pending: tuple[int, int, Any] | None = None
        self._busy: bool = False
        self._closing: bool = False
        self.skipped: int = 0
        """Snapshots replaced before they could be evaluated."""
        self.completed: int = 0
        """Evaluations finished."""
        self._thread: threading.Thread = lifetime.spawn(self._run, "evaluator")

    def submit(self, update: int, step: int, params: Any) -> None:
        """Queue ``params`` for evaluation, replacing any snapshot not yet started."""
        with self._cond:
            if self._pending is not None:
                self.skipped += 1
            self._pending = (update, step, params)
            self._cond.notify_all()

    def finish(self, timeout: float = 600.0) -> None:
        """Evaluate whatever is pending, then stop the thread."""
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._closing:
                    if self._lifetime.stopped:
                        return
                    self._cond.wait(0.1)
                if self._pending is None:
                    return
                update, step, params = self._pending
                self._pending = None
                self._busy = True
            metrics = self._eval_fn(update, params)
            self._sink(update, step, metrics)
            with self._cond:
                self._busy = False
                self.completed += 1
