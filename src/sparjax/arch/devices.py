"""Device selection for actors and the learner."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from typing import Any, Literal

import jax

__all__ = ["Device", "DeviceKind", "resolve_device"]

Device = Any
"""Alias for ``jax.Device`` (whose type jax does not expose to static checkers)."""

DeviceKind = Literal["cpu", "default"]


def resolve_device(kind: DeviceKind, index: int = 0) -> Device:
    """Return a JAX device.

    Args:
        kind: ``"cpu"`` for a host CPU device, or ``"default"`` for the default backend
            (the GPU when a CUDA jaxlib is installed, otherwise the CPU).
        index: Which device of that kind.
    """
    devices = jax.devices("cpu") if kind == "cpu" else jax.devices()
    if not 0 <= index < len(devices):
        raise ValueError(f"no {kind} device with index {index}; have {len(devices)}")
    return devices[index]
