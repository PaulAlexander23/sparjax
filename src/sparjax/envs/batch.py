"""Batches of AEC environments behind one stepping interface.

:class:`AECBatch` is the boundary between the training loop and the environments. Code
above it never touches an ``AECEnv`` directly. Today there is :class:`InProcessAECBatch`;
a subprocess-backed batch (M6) will implement the same protocol, and each call is
designed to cost one message per worker.

Conventions:

* A *seat* is an agent's index in ``env.possible_agents`` (0 or 1 for two players).
* ``reset``/``step`` take the ids of the envs to act on and return a :class:`BatchView`
  describing only those envs, in the same order.
* After every call, each returned env is either at a decision point of a live agent
  (``to_move >= 0``) or the game is over (``done``). Dead-agent bookkeeping required by
  the AEC API (stepping ``None``) is handled inside the batch.
"""
# PettingZoo's AECEnv attributes are only partially typed.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeStubs=false

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from jaxtyping import Bool, Float, Int, Int8
from pettingzoo import AECEnv

__all__ = ["AECBatch", "BatchView", "InProcessAECBatch"]

EnvFactory = Callable[[], AECEnv[Any, Any, Any]]


@dataclass(frozen=True, slots=True)
class BatchView:
    """The state of a subset of envs after ``reset`` or ``step``.

    Attributes:
        env_ids: Which envs this view describes.
        to_move: Seat whose turn it is, or ``-1`` if the game is over.
        observations: Observation of the seat to move (zeros if the game is over).
        action_masks: Legal actions of the seat to move (zeros if the game is over).
        rewards: Reward each seat received during this call.
        done: The game ended during this call (termination or truncation).
        truncated: The game ended by truncation rather than termination.
        info_states: Information-state key of the seat to move (``""`` if absent/over).
    """

    env_ids: Int[np.ndarray, " n"]
    to_move: Int[np.ndarray, " n"]
    observations: Float[np.ndarray, "n obs_dim"]
    action_masks: Int8[np.ndarray, "n num_actions"]
    rewards: Float[np.ndarray, "n num_seats"]
    done: Bool[np.ndarray, " n"]
    truncated: Bool[np.ndarray, " n"]
    info_states: list[str]
    final_observations: dict[int, np.ndarray] = field(default_factory=dict[int, np.ndarray])
    """For rows whose game ended by truncation: each seat's final observation,
    ``[num_seats, *obs_shape]``, used to bootstrap the value of the cut-off position."""


class AECBatch(Protocol):
    """A fixed-size collection of AEC environments stepped together."""

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        ...

    @property
    def num_seats(self) -> int:
        """Number of agents per game."""
        ...

    @property
    def obs_shape(self) -> tuple[int, ...]:
        """Shape of one observation tensor."""
        ...

    @property
    def num_actions(self) -> int:
        """Size of the (shared, discrete) action space."""
        ...

    @property
    def obs_dtype(self) -> np.dtype[Any]:
        """Dtype of observations: ``bool`` for boolean planes (e.g. Go), else ``float32``."""
        ...

    def reset(self, env_ids: Int[np.ndarray, " n"], seeds: Int[np.ndarray, " n"]) -> BatchView:
        """Start new games in ``env_ids`` with the given seeds."""
        ...

    def step(self, env_ids: Int[np.ndarray, " n"], actions: Int[np.ndarray, " n"]) -> BatchView:
        """Apply one action (by the seat to move) in each of ``env_ids``."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


def _split_observation(obs: Any, info: dict[str, Any], num_actions: int) -> tuple[Any, Any]:
    """Return ``(observation_tensor, action_mask)`` from either PettingZoo convention."""
    if isinstance(obs, dict) and "observation" in obs:
        tensor = obs["observation"]
        mask = obs.get("action_mask")
    else:
        tensor, mask = obs, None
    if mask is None:
        mask = info.get("action_mask")
    if mask is None:
        mask = np.ones(num_actions, dtype=np.int8)
    return tensor, mask


class InProcessAECBatch:
    """Runs ``num_envs`` AEC environments sequentially in the calling thread.

    All envs must share agent names, observation shape and a discrete action space of
    the same size.

    Args:
        env_fn: Zero-argument factory that builds one environment.
        num_envs: How many environments to create.
    """

    def __init__(self, env_fn: EnvFactory, num_envs: int) -> None:
        """Create the environments and read their spaces."""
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self._envs: list[AECEnv[Any, Any, Any]] = [env_fn() for _ in range(num_envs)]
        first = self._envs[0]
        self._agents: list[Any] = list(first.possible_agents)
        self._seat: dict[Any, int] = {a: i for i, a in enumerate(self._agents)}
        action_space: Any = first.action_space(self._agents[0])
        self._num_actions: int = int(action_space.n)
        obs_space: Any = first.observation_space(self._agents[0])
        if hasattr(obs_space, "spaces") and "observation" in obs_space.spaces:
            obs_space = obs_space.spaces["observation"]
        self._obs_shape: tuple[int, ...] = tuple(obs_space.shape)
        # Keep boolean planes compact (4x less data to copy or send between processes);
        # the network casts to float.
        self._obs_dtype: np.dtype[Any] = np.dtype(
            np.bool_ if np.dtype(obs_space.dtype) == np.bool_ else np.float32
        )

    # ------------------------------------------------------------------ properties
    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return len(self._envs)

    @property
    def num_seats(self) -> int:
        """Number of agents per game."""
        return len(self._agents)

    @property
    def obs_shape(self) -> tuple[int, ...]:
        """Shape of one observation tensor."""
        return self._obs_shape

    @property
    def num_actions(self) -> int:
        """Size of the discrete action space."""
        return self._num_actions

    @property
    def obs_dtype(self) -> np.dtype[Any]:
        """Dtype of observations."""
        return self._obs_dtype

    @property
    def envs(self) -> Sequence[AECEnv[Any, Any, Any]]:
        """The underlying environments (for inspection and tests)."""
        return self._envs

    # ------------------------------------------------------------------ API
    def reset(self, env_ids: Int[np.ndarray, " n"], seeds: Int[np.ndarray, " n"]) -> BatchView:
        """Start new games in ``env_ids`` with the given seeds."""
        env_ids = np.asarray(env_ids, dtype=np.int64)
        seeds = np.asarray(seeds, dtype=np.int64)
        if env_ids.shape != seeds.shape:
            raise ValueError("env_ids and seeds must have the same shape")
        view = self._empty_view(env_ids)
        for row, (i, seed) in enumerate(zip(env_ids, seeds, strict=True)):
            env = self._envs[i]
            env.reset(seed=int(seed))
            self._collect_rewards(env, view.rewards[row])
            self._settle(env, view, row)
        return view

    def step(self, env_ids: Int[np.ndarray, " n"], actions: Int[np.ndarray, " n"]) -> BatchView:
        """Apply one action by the seat to move in each of ``env_ids``."""
        env_ids = np.asarray(env_ids, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64)
        if env_ids.shape != actions.shape:
            raise ValueError("env_ids and actions must have the same shape")
        view = self._empty_view(env_ids)
        for row, (i, action) in enumerate(zip(env_ids, actions, strict=True)):
            env = self._envs[i]
            if not env.agents or self._game_over(env):
                raise RuntimeError(f"env {i} is finished; reset it before stepping")
            env.step(int(action))
            self._collect_rewards(env, view.rewards[row])
            self._settle(env, view, row)
        return view

    def close(self) -> None:
        """Close all environments."""
        for env in self._envs:
            env.close()

    # ------------------------------------------------------------------ internals
    def _empty_view(self, env_ids: np.ndarray) -> BatchView:
        n = env_ids.shape[0]
        return BatchView(
            env_ids=env_ids,
            to_move=np.full(n, -1, dtype=np.int64),
            observations=np.zeros((n, *self._obs_shape), dtype=self._obs_dtype),
            action_masks=np.zeros((n, self._num_actions), dtype=np.int8),
            rewards=np.zeros((n, self.num_seats), dtype=np.float64),
            done=np.zeros(n, dtype=bool),
            truncated=np.zeros(n, dtype=bool),
            info_states=[""] * n,
        )

    def _collect_rewards(self, env: AECEnv[Any, Any, Any], out: np.ndarray) -> None:
        # ``env.rewards`` holds the rewards produced by the most recent step only.
        for agent, r in env.rewards.items():
            out[self._seat[agent]] += float(r)

    @staticmethod
    def _game_over(env: AECEnv[Any, Any, Any]) -> bool:
        return all(env.terminations[a] or env.truncations[a] for a in env.agents)

    def _settle(self, env: AECEnv[Any, Any, Any], view: BatchView, row: int) -> None:
        """Advance past dead agents, then record whose turn it is or that the game ended."""
        while env.agents and not self._game_over(env):
            agent = env.agent_selection
            if not (env.terminations[agent] or env.truncations[agent]):
                break
            env.step(None)  # an individual agent left early; AEC requires a dead step
            self._collect_rewards(env, view.rewards[row])

        if not env.agents or self._game_over(env):
            view.done[row] = True
            view.truncated[row] = any(env.truncations.values()) and not any(
                env.terminations.values()
            )
            if view.truncated[row]:
                finals = np.zeros((self.num_seats, *self._obs_shape), dtype=self._obs_dtype)
                for agent in env.agents:
                    obs, _ = _split_observation(env.observe(agent), {}, self._num_actions)
                    finals[self._seat[agent]] = np.asarray(obs, self._obs_dtype).reshape(
                        self._obs_shape
                    )
                view.final_observations[row] = finals
            while env.agents:  # drain dead steps so the env is cleanly finished
                env.step(None)
            return

        agent = env.agent_selection
        obs, mask = _split_observation(env.observe(agent), env.infos[agent], self._num_actions)
        view.to_move[row] = self._seat[agent]
        view.observations[row] = np.asarray(obs, dtype=self._obs_dtype).reshape(self._obs_shape)
        view.action_masks[row] = np.asarray(mask, dtype=np.int8)
        view.info_states[row] = str(env.infos[agent].get("info_state", ""))
