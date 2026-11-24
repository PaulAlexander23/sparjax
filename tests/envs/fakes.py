"""A tiny, fully controllable AEC game for testing batching and the opponent loop."""

from typing import Any, Literal, cast, final, override

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv


@final
class ScriptedGame(AECEnv[str, Any, int | None]):
    """Players alternate; every move by the mover gives it ``+(a+1)`` and the others ``-(a+1)``.

    The game ends after ``length`` moves (by termination, or truncation if ``truncate``).
    Observation: ``[moves_so_far, seat_of_observer]``.

    Args:
        length: Number of moves in a game (0 means the game is over at reset).
        first: Seat that moves first.
        mask_mode: Where the action mask lives: ``"obs"`` (dict obs), ``"info"`` or ``"none"``.
        reset_reward: Reward given to seat 0 (and its negation to seat 1) at reset.
        num_agents_: Number of players.
        truncate: End by truncation instead of termination.
        legal: Legal actions (others masked out).
    """

    def __init__(
        self,
        length: int = 3,
        first: int = 0,
        mask_mode: Literal["obs", "info", "none"] = "obs",
        reset_reward: float = 0.0,
        num_agents_: int = 2,
        truncate: bool = False,
        legal: tuple[int, ...] = (0, 1, 2),
    ) -> None:
        super().__init__()
        self.metadata = {"name": "scripted_game_v0", "render_modes": [], "is_parallelizable": False}
        self.length = length
        self.first = first
        self.mask_mode = mask_mode
        self.reset_reward = reset_reward
        self.truncate = truncate
        self.legal = legal
        self.possible_agents = [f"p{i}" for i in range(num_agents_)]
        box = spaces.Box(0, 100, shape=(2,), dtype=np.float32)
        mask_box = spaces.Box(0, 1, shape=(3,), dtype=np.int8)
        obs_space: Any = (
            spaces.Dict({"observation": box, "action_mask": mask_box})
            if mask_mode == "obs"
            else box
        )
        self.observation_spaces = cast(
            "dict[str, spaces.Space[Any]]", dict.fromkeys(self.possible_agents, obs_space)
        )
        self.action_spaces = cast(
            "dict[str, spaces.Space[int | None]]",
            {a: spaces.Discrete(3) for a in self.possible_agents},
        )
        self.moves = 0
        self.seeds: list[int | None] = []
        self.history: list[tuple[str, int]] = []

    @override
    def observation_space(self, agent: str) -> Any:
        return self.observation_spaces[agent]

    @override
    def action_space(self, agent: str) -> Any:
        return self.action_spaces[agent]

    def _mask(self, agent: str) -> np.ndarray:
        mask = np.zeros(3, dtype=np.int8)
        if agent == self.agent_selection and not self._over():
            mask[list(self.legal)] = 1
        return mask

    def _over(self) -> bool:
        return self.moves >= self.length

    @override
    def observe(self, agent: str) -> Any:
        obs = np.array([self.moves, self.possible_agents.index(agent)], dtype=np.float32)
        if self.mask_mode == "obs":
            return {"observation": obs, "action_mask": self._mask(agent)}
        return obs

    def _refresh(self) -> None:
        over = self._over()
        for a in self.agents:
            self.terminations[a] = over and not self.truncate
            self.truncations[a] = over and self.truncate
            info: dict[str, Any] = {"info_state": f"{a}:{self.moves}"}
            if self.mask_mode == "info":
                info["action_mask"] = self._mask(a)
            self.infos[a] = info

    @override
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        self.seeds.append(seed)
        self.agents = self.possible_agents[:]
        self.moves = 0
        self.history = []
        self.agent_selection = self.possible_agents[self.first]
        self.rewards = dict.fromkeys(self.agents, 0.0)
        if self.reset_reward:
            self.rewards[self.agents[0]] = self.reset_reward
            self.rewards[self.agents[1]] = -self.reset_reward
        self._cumulative_rewards = dict(self.rewards)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {a: {} for a in self.agents}
        self._refresh()

    @override
    def step(self, action: int | None) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        assert action is not None
        assert action in self.legal, f"illegal action {action}"
        self.history.append((agent, action))
        self._cumulative_rewards[agent] = 0.0
        self.rewards = {a: (action + 1.0 if a == agent else -(action + 1.0)) for a in self.agents}
        self.moves += 1
        idx = self.possible_agents.index(agent)
        self.agent_selection = self.possible_agents[(idx + 1) % len(self.possible_agents)]
        self._refresh()
        self._accumulate_rewards()
