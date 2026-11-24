import threading
import time

import pytest

from sparjax.arch import Lifetime
from sparjax.arch.evaluator import AsyncEvaluator


def test_newest_snapshot_wins_and_final_is_evaluated() -> None:
    life = Lifetime()
    gate = threading.Event()
    seen: list[tuple[int, int, float]] = []

    def slow_eval(_update: int, params: float) -> dict[str, float]:
        gate.wait(2)
        return {"p": params}

    ev = AsyncEvaluator(slow_eval, lambda u, s, m: seen.append((u, s, m["p"])), life)
    ev.submit(1, 10, 1.0)
    time.sleep(0.05)  # evaluator picks up update 1 and blocks on the gate
    ev.submit(2, 20, 2.0)
    ev.submit(3, 30, 3.0)  # replaces 2
    gate.set()
    ev.finish()
    assert [u for u, _, _ in seen] == [1, 3]
    assert seen[-1] == (3, 30, 3.0)
    assert ev.skipped == 1
    assert ev.completed == 2
    assert not life.join()


def test_evaluator_errors_propagate() -> None:
    life = Lifetime()

    def bad(_update: int, _params: object) -> dict[str, float]:
        raise ValueError("eval broke")

    ev = AsyncEvaluator(bad, lambda u, s, m: None, life)
    ev.submit(1, 1, None)
    ev.finish()
    with pytest.raises(RuntimeError, match="worker thread failed"):
        life.raise_if_failed()


def test_evaluator_exits_on_stop() -> None:
    life = Lifetime()
    ev = AsyncEvaluator(lambda u, p: {}, lambda u, s, m: None, life)
    life.stop()
    ev.finish(timeout=2)
    assert not life.join(timeout=2)
