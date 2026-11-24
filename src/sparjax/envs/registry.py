"""Name environments by id so they can be built anywhere, including worker processes.

Ids have the form ``<backend>:<name>``:

* ``openspiel:<game>``: :class:`~sparjax.envs.openspiel_aec.OpenSpielAECEnv`, e.g.
  ``openspiel:leduc_poker``.
* ``pettingzoo:<registered id>``: any env in PettingZoo's AEC registry, built with
  ``pettingzoo.make("aec", ...)``, e.g. ``pettingzoo:classic/go-v5``.

An :class:`EnvSpec` is a small frozen dataclass and pickles cleanly, which subprocess
workers need. Lambdas and closures do not.
"""

# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pettingzoo import AECEnv

__all__ = ["EnvSpec", "parse_env_id"]

_BACKENDS = ("openspiel", "pettingzoo")


def parse_env_id(env_id: str) -> tuple[str, str]:
    """Split ``"backend:name"`` and validate the backend."""
    backend, sep, name = env_id.partition(":")
    if not sep or not name or backend not in _BACKENDS:
        raise ValueError(
            f"env id must look like 'openspiel:<game>' or 'pettingzoo:<id>', got {env_id!r}"
        )
    return backend, name


@dataclass(frozen=True)
class EnvSpec:
    """A picklable recipe for one environment.

    Attributes:
        id: ``"openspiel:<game>"`` or ``"pettingzoo:<registered id>"``.
        kwargs: Keyword arguments for the environment constructor.
        max_steps: If set, truncate each game after this many actions (any agent).
    """

    id: str
    kwargs: Mapping[str, Any] = field(default_factory=dict[str, Any])
    max_steps: int | None = None

    def __post_init__(self) -> None:
        """Validate the id eagerly so mistakes surface before any worker starts."""
        parse_env_id(self.id)

    @property
    def backend(self) -> str:
        """``"openspiel"`` or ``"pettingzoo"``."""
        return parse_env_id(self.id)[0]

    @property
    def name(self) -> str:
        """The part after the colon."""
        return parse_env_id(self.id)[1]

    @property
    def openspiel_game(self) -> str | None:
        """The OpenSpiel game name, or ``None`` for non-OpenSpiel envs."""
        return self.name if self.backend == "openspiel" else None

    def __call__(self) -> AECEnv[Any, Any, Any]:
        """Build a fresh environment."""
        env = self._build()
        if self.max_steps is not None:
            from sparjax.envs.wrappers import MaxStepsWrapper

            env = MaxStepsWrapper(env, self.max_steps)
        return env

    def _build(self) -> AECEnv[Any, Any, Any]:
        if self.backend == "openspiel":
            from sparjax.envs.openspiel_aec import OpenSpielAECEnv  # optional dependency

            return OpenSpielAECEnv(self.name, **dict(self.kwargs))
        # pygame (needed by some PettingZoo classic envs) prints a banner on import.
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pettingzoo

        env = pettingzoo.make("aec", self.name, **dict(self.kwargs))
        assert isinstance(env, AECEnv)
        return env
