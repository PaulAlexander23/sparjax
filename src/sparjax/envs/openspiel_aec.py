"""PettingZoo AEC wrapper around two-player, turn-based OpenSpiel games.

This is the only module (together with :mod:`sparjax.opponents.cfr`) that talks to
``pyspiel`` directly. ``pyspiel`` ships without type stubs, so the unknown-type checks
are relaxed for this file only.

Note:
    ``OpenSpielAECEnv("leduc_poker")`` is **OpenSpiel's** Leduc poker. It is *not*
    PettingZoo's ``classic/leduc_holdem`` (which wraps RLCard and has different betting
    rules, action IDs and observations). Results are not comparable between the two.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false

from __future__ import annotations

from typing import Any, Final, cast, final, override

import numpy as np
import pyspiel
from gymnasium import spaces
from pettingzoo import AECEnv

from sparjax.envs.types import AgentID, Observation

__all__ = ["OpenSpielAECEnv"]

_NUM_PLAYERS: Final = 2


@final
class OpenSpielAECEnv(AECEnv[AgentID, Observation, int | None]):
    """A two-player, zero-sum, sequential OpenSpiel game exposed through the AEC API.

    Observations are dicts with:

    * ``"observation"``: the acting agent's ``information_state_tensor`` (float32).
    * ``"action_mask"``: 1 for legal actions, 0 otherwise (all zeros when it is not
      that agent's turn), following the PettingZoo classic convention.

    ``infos[agent]["info_state"]`` holds that agent's ``information_state_string``. It is
    the key used by tabular policies such as CFR.

    Chance nodes (e.g. card deals) are resolved inside the env with a NumPy generator
    seeded by :meth:`reset`, so the same seed gives the same deals.

    An illegal action raises :class:`ValueError` instead of silently ending the game.
    Training code should always respect the action mask.

    Args:
        game_name: OpenSpiel short name, e.g. ``"kuhn_poker"`` or ``"leduc_poker"``.
        render_mode: ``None`` or ``"ansi"``.
        **game_params: Extra OpenSpiel game parameters.

    Raises:
        ValueError: If the game is not a two-player, zero-sum, sequential game with an
            information-state tensor.
    """

    def __init__(self, game_name: str, render_mode: str | None = None, **game_params: Any) -> None:
        """Load and validate the OpenSpiel game."""
        super().__init__()
        game = pyspiel.load_game(game_name, game_params)
        game_type = game.get_type()
        if game.num_players() != _NUM_PLAYERS:
            msg = f"{game_name}: expected {_NUM_PLAYERS} players, got {game.num_players()}"
            raise ValueError(msg)
        if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise ValueError(f"{game_name}: only sequential (turn-based) games are supported")
        if game_type.utility != pyspiel.GameType.Utility.ZERO_SUM:
            raise ValueError(f"{game_name}: only zero-sum games are supported")
        if not game_type.provides_information_state_tensor:
            raise ValueError(f"{game_name}: game does not provide an information_state_tensor")
        if render_mode not in (None, "ansi"):
            raise ValueError(f"unsupported render_mode {render_mode!r}")

        self.game_name: Final[str] = game_name
        self.game: Final = game
        self.render_mode: str | None = render_mode
        self.metadata = {
            "name": f"openspiel_{game_name}_v0",
            "render_modes": ["ansi"],
            "is_parallelizable": False,
        }

        self.possible_agents = [f"player_{i}" for i in range(_NUM_PLAYERS)]
        self._agent_to_player: Final = {a: i for i, a in enumerate(self.possible_agents)}
        self._num_actions: Final[int] = game.num_distinct_actions()
        obs_shape = tuple(game.information_state_tensor_shape())
        obs_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32),
                "action_mask": spaces.Box(0, 1, shape=(self._num_actions,), dtype=np.int8),
            }
        )
        self.observation_spaces = cast(
            "dict[AgentID, spaces.Space[Observation]]",
            dict.fromkeys(self.possible_agents, obs_space),
        )
        self.action_spaces = cast(
            "dict[AgentID, spaces.Space[int | None]]",
            {a: spaces.Discrete(self._num_actions) for a in self.possible_agents},
        )

        self._rng = np.random.default_rng()
        self._state = game.new_initial_state()

    # ------------------------------------------------------------------ properties
    @property
    def game_state(self) -> Any:
        """The underlying ``pyspiel.State``. Treat as read-only."""
        return self._state

    def player_index(self, agent: AgentID) -> int:
        """OpenSpiel player id for a PettingZoo agent name."""
        return self._agent_to_player[agent]

    # ------------------------------------------------------------------ spaces
    @override
    def observation_space(self, agent: AgentID) -> spaces.Space[Any]:
        return self.observation_spaces[agent]

    @override
    def action_space(self, agent: AgentID) -> spaces.Space[Any]:
        return self.action_spaces[agent]

    # ------------------------------------------------------------------ AEC API
    @override
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self.game.new_initial_state()
        self._resolve_chance()

        self.agents = self.possible_agents[:]
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {a: {} for a in self.agents}
        self._sync_after_transition()

    @override
    def step(self, action: int | None) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        if action is None:
            raise ValueError(f"{agent} is still in the game; action must not be None")
        action = int(action)
        if action not in self._state.legal_actions():
            legal = self._state.legal_actions()
            raise ValueError(f"illegal action {action} for {agent}; legal actions: {legal}")

        self._cumulative_rewards[agent] = 0.0
        self._state.apply_action(action)
        self._resolve_chance()
        self._sync_after_transition()
        self._accumulate_rewards()

    @override
    def observe(self, agent: AgentID) -> Observation:
        player = self._agent_to_player[agent]
        obs = np.asarray(self._state.information_state_tensor(player), dtype=np.float32)
        mask = np.zeros(self._num_actions, dtype=np.int8)
        if not self._state.is_terminal() and self._state.current_player() == player:
            mask[self._state.legal_actions()] = 1
        return {"observation": obs, "action_mask": mask}

    @override
    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return str(self._state)
        return None

    @override
    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ internals
    def _resolve_chance(self) -> None:
        while self._state.is_chance_node():
            outcomes, probs = zip(*self._state.chance_outcomes(), strict=True)
            idx = self._rng.choice(len(outcomes), p=np.asarray(probs, dtype=np.float64))
            self._state.apply_action(outcomes[idx])

    def _sync_after_transition(self) -> None:
        """Refresh rewards, terminal flags, infos and agent_selection from the state."""
        terminal = self._state.is_terminal()
        returns = self._state.returns() if terminal else [0.0] * _NUM_PLAYERS
        for agent in self.agents:
            player = self._agent_to_player[agent]
            self.rewards[agent] = float(returns[player])
            self.terminations[agent] = terminal
            self.infos[agent] = {"info_state": self._state.information_state_string(player)}
        if not terminal:
            self.agent_selection = self.possible_agents[self._state.current_player()]
        # On a terminal state agent_selection keeps the last mover; PettingZoo's
        # dead-step protocol then cycles through the remaining agents.
