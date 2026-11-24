"""Versioned parameter broadcast from the learner to actor threads.

Stoix's ``ParameterServer`` (Apache-2.0, see ``NOTICE``) keeps one size-1 queue per actor.
This version keeps a single *latest* slot with a version number instead, which makes the
allowed staleness explicit: an actor about to collect rollout ``r`` asks for parameters
with ``version >= r - max_lag``.

* ``max_lag = 0``: strictly on-policy. Acting and learning alternate.
* ``max_lag = 1``: classic Sebulba. Actors collect rollout ``r`` with the parameters from
  update ``r - 1`` while the learner is busy with update ``r``.

Parameters are placed on each actor's device once per publish, not once per actor.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import jax

from sparjax.arch.devices import Device
from sparjax.arch.lifetime import Lifetime

__all__ = ["ParameterServer"]


class ParameterServer:
    """Holds the latest parameters (per actor device) and their version."""

    def __init__(self, devices: Sequence[Device], lifetime: Lifetime) -> None:
        """Create an empty server for actors running on ``devices``."""
        self._devices: list[Device] = list(dict.fromkeys(devices))
        self._lifetime: Lifetime = lifetime
        self._cond: threading.Condition = threading.Condition()
        self._version: int = -1
        self._params: dict[Device, Any] = {}

    @property
    def version(self) -> int:
        """Version of the latest published parameters (``-1`` before the first publish)."""
        with self._cond:
            return self._version

    def publish(self, params: Any, version: int) -> None:
        """Make ``params`` available to actors as ``version`` (must increase)."""
        placed = {d: jax.device_put(params, d) for d in self._devices}
        with self._cond:
            if version <= self._version:
                raise ValueError(f"version {version} is not newer than {self._version}")
            self._params = placed
            self._version = version
            self._cond.notify_all()

    def get(self, device: Device, min_version: int, poll: float = 0.1) -> tuple[Any, int]:
        """Block until parameters with ``version >= min_version`` exist; return them.

        Raises:
            StopRequestedError: If shutdown is requested while waiting.
        """
        with self._cond:
            while self._version < min_version:
                self._lifetime.check()
                self._cond.wait(poll)
            return self._params[device], self._version
