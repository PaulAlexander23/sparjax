"""AEC wrappers."""
# PettingZoo's wrappers and env attributes are only partially typed.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false

from __future__ import annotations

from typing import Any, override

from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper

__all__ = ["MaxStepsWrapper"]


class MaxStepsWrapper(BaseWrapper[Any, Any, Any]):
    """Truncate a game after ``max_steps`` actions (by any agent).

    When the limit is reached and the game has not ended on its own, every remaining
    agent is marked truncated with zero reward, which the batch reports as
    ``done & truncated``. The learner then bootstraps the value of the final position
    instead of treating it as a loss or a draw.
    """

    def __init__(self, env: AECEnv[Any, Any, Any], max_steps: int) -> None:
        """Wrap ``env``."""
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        super().__init__(env)
        self.max_steps: int = max_steps
        self._steps: int = 0

    @override
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        self._steps = 0
        super().reset(seed=seed, options=options)

    @override
    def step(self, action: Any) -> None:
        agent = self.agent_selection
        live = not (self.terminations[agent] or self.truncations[agent])
        super().step(action)
        if not live:
            return
        self._steps += 1
        if self._steps >= self.max_steps and self.agents:
            raw = self.env.unwrapped
            if not all(raw.terminations[a] or raw.truncations[a] for a in raw.agents):
                for a in raw.agents:
                    raw.truncations[a] = True
