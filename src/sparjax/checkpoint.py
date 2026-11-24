"""Save and load network parameters (plus the run config) as ``.npz`` files."""
# flax.nnx pure-dict helpers are only partially typed.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import nnx

from sparjax.agents.ppo.networks import ActorCritic, Params, init_actor_critic

__all__ = ["Checkpoint", "config_to_dict", "load_checkpoint", "save_checkpoint"]

_SEP = "/"


def _flatten(tree: Mapping[Any, Any], prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in tree.items():
        name = f"{prefix}{_SEP}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten(value, name))
        else:
            out[name] = np.asarray(value)
    return out


def _unflatten(flat: Mapping[str, np.ndarray]) -> dict[Any, Any]:
    root: dict[Any, Any] = {}
    for name, value in flat.items():
        node = root
        *parents, leaf = name.split(_SEP)
        for part in parents:
            node = node.setdefault(int(part) if part.isdigit() else part, {})
        node[int(leaf) if leaf.isdigit() else leaf] = value
    return root


def config_to_dict(config: Any) -> dict[str, Any]:
    """JSON-friendly dict of a (nested) dataclass config."""
    return json.loads(json.dumps(dataclasses.asdict(config), default=str))


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """A loaded checkpoint."""

    graphdef: nnx.GraphDef[ActorCritic]
    params: Params
    meta: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    params: Params,
    *,
    obs_shape: int | tuple[int, ...],
    num_actions: int,
    hidden: tuple[int, ...],
    conv_channels: tuple[int, ...] = (),
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``params`` and enough metadata to rebuild the network."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "format": "sparjax.checkpoint",
        "version": 1,
        "obs_shape": [obs_shape] if isinstance(obs_shape, int) else list(obs_shape),
        "conv_channels": list(conv_channels),
        "num_actions": num_actions,
        "hidden": list(hidden),
        "meta": dict(meta or {}),
    }
    arrays = _flatten(nnx.to_pure_dict(jax.device_get(params)))
    with path.open("wb") as f:
        payload: dict[str, Any] = {"__header__": np.array(json.dumps(header)), **arrays}
        np.savez_compressed(f, **payload)
    return path


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Rebuild the network written by :func:`save_checkpoint`."""
    with np.load(Path(path), allow_pickle=False) as data:
        header: dict[str, Any] = json.loads(str(data["__header__"]))
        if header.get("format") != "sparjax.checkpoint":
            raise ValueError(f"{path} is not a sparjax checkpoint")
        flat = {k: np.asarray(data[k]) for k in data.files if k != "__header__"}
    graphdef, params = init_actor_critic(
        tuple(header["obs_shape"]),
        header["num_actions"],
        tuple(header["hidden"]),
        seed=0,
        conv_channels=tuple(header["conv_channels"]),
    )
    nnx.replace_by_pure_dict(params, _unflatten(flat))
    return Checkpoint(graphdef=graphdef, params=params, meta=header["meta"])
