"""An :class:`~sparjax.envs.batch.AECBatch` whose environments live in worker processes.

Python env stepping holds the GIL, so actor threads alone cannot step expensive
environments in parallel. :class:`SubprocessAECBatch` spreads the envs over ``K`` worker
processes. Each worker owns a contiguous slice of env ids and runs an
:class:`~sparjax.envs.batch.InProcessAECBatch` over it. A ``reset``/``step`` call sends one
message to every worker that owns an affected env, lets them all work in parallel, then
gathers the replies and reassembles them in the caller's order.

Workers are started with the ``spawn`` method: forking a process that already runs JAX
and threads is unsafe. The env factory must therefore be picklable, e.g. an
:class:`~sparjax.envs.registry.EnvSpec` (a lambda or closure is not).

The opponent stays in the actor process. Workers only step environments.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
from jaxtyping import Int

from sparjax.envs.batch import BatchView, InProcessAECBatch

__all__ = ["SubprocessAECBatch"]

EnvFactory = Callable[[], Any]


# Runs in spawned child processes, which coverage cannot see; exercised by the subprocess tests.
def _worker(conn: Connection, env_fn: EnvFactory, num_envs: int) -> None:  # pragma: no cover
    """Worker loop: own ``num_envs`` envs and answer commands until told to close."""
    # Keep numerical libraries in workers single-threaded; parallelism comes from workers.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        batch = InProcessAECBatch(env_fn, num_envs)
        info = (batch.num_seats, batch.obs_shape, batch.num_actions, batch.obs_dtype.str)
        conn.send(("ok", info))
    except BaseException:  # pragma: no cover - reported to the parent
        conn.send(("error", traceback.format_exc()))
        return
    while True:
        try:
            cmd, a, b = conn.recv()
        except EOFError:  # parent went away
            break
        try:
            if cmd == "reset":
                conn.send(("ok", batch.reset(a, b)))
            elif cmd == "step":
                conn.send(("ok", batch.step(a, b)))
            elif cmd == "close":
                batch.close()
                conn.send(("ok", None))
                break
            else:
                conn.send(("error", f"unknown command {cmd!r}"))
        except BaseException:
            conn.send(("error", traceback.format_exc()))
    conn.close()


class SubprocessAECBatch:
    """``num_envs`` AEC environments spread over ``num_workers`` processes.

    Args:
        env_fn: Picklable zero-argument factory for one environment.
        num_envs: Total number of environments.
        num_workers: Number of worker processes (each gets a near-equal slice).
        start_timeout: Seconds to wait for each worker to build its envs.
    """

    def __init__(
        self, env_fn: EnvFactory, num_envs: int, num_workers: int, start_timeout: float = 120.0
    ) -> None:
        """Start the workers and read the env spaces."""
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if not 1 <= num_workers <= num_envs:
            raise ValueError("num_workers must be between 1 and num_envs")
        ctx = mp.get_context("spawn")
        sizes = [len(c) for c in np.array_split(np.arange(num_envs), num_workers)]
        self._starts: np.ndarray = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        self._owner: np.ndarray = np.repeat(np.arange(num_workers), sizes)
        self._num_envs: int = num_envs
        self._conns: list[Any] = []  # multiprocessing Connection objects
        self._procs: list[Any] = []
        self._closed: bool = False
        for size in sizes:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, env_fn, size), daemon=True)
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        infos = [self._recv(i, start_timeout) for i in range(num_workers)]
        if len(set(infos)) != 1:
            self.close()
            raise ValueError(f"workers disagree on env spaces: {infos}")
        num_seats, obs_shape, num_actions, dtype = infos[0]
        self._num_seats: int = int(num_seats)
        self._obs_shape: tuple[int, ...] = tuple(obs_shape)
        self._num_actions: int = int(num_actions)
        self._obs_dtype: np.dtype[Any] = np.dtype(dtype)

    # ------------------------------------------------------------------ properties
    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._num_envs

    @property
    def num_workers(self) -> int:
        """Number of worker processes."""
        return len(self._procs)

    @property
    def num_seats(self) -> int:
        """Number of agents per game."""
        return int(self._num_seats)

    @property
    def obs_shape(self) -> tuple[int, ...]:
        """Shape of one observation tensor."""
        return tuple(self._obs_shape)

    @property
    def num_actions(self) -> int:
        """Size of the discrete action space."""
        return int(self._num_actions)

    @property
    def obs_dtype(self) -> np.dtype[Any]:
        """Dtype of observations."""
        return self._obs_dtype

    # ------------------------------------------------------------------ API
    def reset(self, env_ids: Int[np.ndarray, " n"], seeds: Int[np.ndarray, " n"]) -> BatchView:
        """Start new games in ``env_ids`` with the given seeds."""
        return self._call("reset", env_ids, seeds)

    def step(self, env_ids: Int[np.ndarray, " n"], actions: Int[np.ndarray, " n"]) -> BatchView:
        """Apply one action by the seat to move in each of ``env_ids``."""
        return self._call("step", env_ids, actions)

    def close(self, timeout: float = 5.0) -> None:
        """Ask workers to close their envs, then make sure the processes are gone."""
        if self._closed:
            return
        self._closed = True
        for conn in self._conns:
            with contextlib.suppress(BrokenPipeError, OSError):
                conn.send(("close", None, None))
        for i, proc in enumerate(self._procs):
            try:
                if self._conns[i].poll(timeout):
                    self._conns[i].recv()
            except (EOFError, OSError):
                pass
            proc.join(timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout)
            self._conns[i].close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        """Close workers if the user forgot to."""
        with contextlib.suppress(Exception):
            self.close(timeout=1.0)

    # ------------------------------------------------------------------ internals
    def _recv(self, worker: int, timeout: float | None = None) -> Any:
        conn = self._conns[worker]
        if timeout is not None and not conn.poll(timeout):
            raise TimeoutError(f"env worker {worker} did not answer within {timeout} s")
        try:
            status, payload = conn.recv()
        except EOFError as exc:
            raise RuntimeError(f"env worker {worker} died") from exc
        if status != "ok":
            raise RuntimeError(f"env worker {worker} failed:\n{payload}")
        return payload

    def _call(self, cmd: str, env_ids: np.ndarray, values: np.ndarray) -> BatchView:
        env_ids = np.asarray(env_ids, dtype=np.int64)
        values = np.asarray(values, dtype=np.int64)
        if env_ids.shape != values.shape:
            raise ValueError("env_ids and their values must have the same shape")
        if self._closed:
            raise RuntimeError("batch is closed")
        owners = self._owner[env_ids]
        workers = np.unique(owners)
        rows_by_worker = {int(w): np.flatnonzero(owners == w) for w in workers}
        for w, rows in rows_by_worker.items():
            local = env_ids[rows] - self._starts[w]
            self._conns[w].send((cmd, local, values[rows]))
        parts = {w: self._recv(w) for w in rows_by_worker}
        return self._assemble(env_ids, rows_by_worker, parts)

    def _assemble(
        self,
        env_ids: np.ndarray,
        rows_by_worker: dict[int, np.ndarray],
        parts: dict[int, BatchView],
    ) -> BatchView:
        n = env_ids.shape[0]
        first = next(iter(parts.values()))
        to_move = np.empty(n, dtype=first.to_move.dtype)
        obs = np.empty((n, *first.observations.shape[1:]), dtype=first.observations.dtype)
        masks = np.empty((n, *first.action_masks.shape[1:]), dtype=first.action_masks.dtype)
        rewards = np.empty((n, *first.rewards.shape[1:]), dtype=first.rewards.dtype)
        done = np.empty(n, dtype=bool)
        truncated = np.empty(n, dtype=bool)
        info_states = [""] * n
        for w, rows in rows_by_worker.items():
            part = parts[w]
            to_move[rows] = part.to_move
            obs[rows] = part.observations
            masks[rows] = part.action_masks
            rewards[rows] = part.rewards
            done[rows] = part.done
            truncated[rows] = part.truncated
            for row, info in zip(rows, part.info_states, strict=True):
                info_states[row] = info
        return BatchView(env_ids, to_move, obs, masks, rewards, done, truncated, info_states)
