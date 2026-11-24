"""Tabular policies (e.g. CFR output) and an opponent that plays them."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int

from sparjax.opponents.base import OpponentBatch, sample_actions

__all__ = ["TabularPolicy", "TabularPolicyOpponent"]


@dataclass(frozen=True)
class TabularPolicy:
    """A map from information-state string to a distribution over actions.

    Attributes:
        game_name: The OpenSpiel game the policy belongs to.
        num_actions: Size of the action space.
        probs: ``info_state -> probabilities`` (length ``num_actions``).
        metadata: Free-form JSON-serialisable details (solver, iterations, game params).
    """

    game_name: str
    num_actions: int
    probs: Mapping[str, Float[np.ndarray, " num_actions"]]
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        """Validate shapes and that every row is a probability distribution."""
        for key, row in self.probs.items():
            if row.shape != (self.num_actions,):
                raise ValueError(f"row for {key!r} has shape {row.shape}")
            if (row < 0).any() or not np.isclose(row.sum(), 1.0, atol=1e-5):
                raise ValueError(f"row for {key!r} is not a probability distribution")

    def __len__(self) -> int:
        """Number of information states in the table."""
        return len(self.probs)

    def save(self, path: str | Path) -> Path:
        """Write the policy to a compressed ``.npz`` file and return its path."""
        path = Path(path)
        keys = list(self.probs)
        matrix = (
            np.stack([self.probs[k] for k in keys]).astype(np.float64)
            if keys
            else np.zeros((0, self.num_actions))
        )
        header = {
            "format": "sparjax.tabular_policy",
            "version": 1,
            "game_name": self.game_name,
            "num_actions": self.num_actions,
            "metadata": dict(self.metadata),
        }
        with path.open("wb") as f:
            np.savez_compressed(
                f, header=np.array(json.dumps(header)), info_states=np.array(keys), probs=matrix
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> TabularPolicy:
        """Read a policy written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            header: dict[str, Any] = json.loads(str(data["header"]))
            if header.get("format") != "sparjax.tabular_policy":
                raise ValueError(f"{path} is not a sparjax tabular policy file")
            keys = [str(k) for k in data["info_states"]]
            matrix = np.asarray(data["probs"], dtype=np.float64)
        return cls(
            game_name=str(header["game_name"]),
            num_actions=int(header["num_actions"]),
            probs={k: matrix[i] for i, k in enumerate(keys)},
            metadata=header.get("metadata", {}),
        )


class TabularPolicyOpponent:
    """Plays a :class:`TabularPolicy` by looking up each game's information state.

    Args:
        policy: The table to play.
        on_missing: What to do for an information state not in the table: ``"error"``
            (default, catches env/policy mismatches early) or ``"uniform"``.
    """

    def __init__(
        self, policy: TabularPolicy, on_missing: Literal["error", "uniform"] = "error"
    ) -> None:
        """Create the opponent."""
        self.policy: TabularPolicy = policy
        self.on_missing: Literal["error", "uniform"] = on_missing

    def act(self, batch: OpponentBatch, rng: np.random.Generator) -> Int[np.ndarray, " batch"]:
        """Sample actions from the table for each game."""
        if batch.info_states is None:
            raise ValueError("TabularPolicyOpponent needs info_states in the batch")
        if batch.action_masks.shape[1] != self.policy.num_actions:
            raise ValueError("action space size does not match the policy")
        probs = np.empty(batch.action_masks.shape, dtype=np.float64)
        for i, key in enumerate(batch.info_states):
            row = self.policy.probs.get(key)
            if row is None:
                if self.on_missing == "error":
                    raise KeyError(f"information state not in policy: {key!r}")
                row = batch.action_masks[i].astype(np.float64)
            probs[i] = row
        return sample_actions(probs, batch.action_masks, rng)
