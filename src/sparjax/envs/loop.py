"""Turn a batch of two-player AEC games into a fixed-shape single-agent stream.

:class:`BatchedOpponentLoop` resolves every opponent move before handing control back
to the learner. From the learner's point of view each env is an ordinary single-agent
MDP: every :meth:`~BatchedOpponentLoop.step` takes one action per env and returns one
observation, reward and done flag per env. That is what lets the PPO learner stay
identical to single-agent Sebulba.

Behaviour (see the project brief, section 4.2):

* On every reset the learner gets a random seat, and the env gets the loop's *current*
  opponent. Switching opponents (:meth:`~BatchedOpponentLoop.set_opponent`) only
  affects games that start afterwards, so each game is played against one opponent.
* Opponent moves are batched: all envs waiting on the same opponent are sent in one
  :meth:`~sparjax.opponents.Opponent.act` call.
* The learner's reward for a step is everything its seat received from its action
  until its next decision point (or the end of the game).
* Finished games are reset automatically. ``dones[i]`` is true when the game in env
  ``i`` ended after the learner's action, and ``observations[i]`` then already belongs
  to the next game (the usual auto-reset convention).
* Rare edge case: a game that ends before the learner has acted at all gives the learner
  no transition to credit. It is reported to the provider and counted in
  :attr:`~BatchedOpponentLoop.skipped_episodes`, then the env is reset.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Bool, Float, Int, Int8

from sparjax.envs.batch import AECBatch, BatchView
from sparjax.opponents.base import Opponent, OpponentBatch, OpponentId, OpponentProvider

__all__ = ["BatchedOpponentLoop", "EpisodeRecord", "LearnerStep"]

_NUM_SEATS = 2


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """Summary of one finished game, from the learner's point of view."""

    env_id: int
    opponent_id: OpponentId
    learner_seat: int
    learner_return: float
    learner_decisions: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class LearnerStep:
    """What the learner sees after :meth:`BatchedOpponentLoop.reset` or ``step``.

    Attributes:
        observations: Observation at the learner's next decision point, per env.
        action_masks: Legal actions at that decision point.
        rewards: Learner reward since its previous action (0 after ``reset``).
        dones: The game ended after the learner's previous action.
        truncated: That ending was a truncation (bootstrap the value) not a termination.
        learner_seats: The learner's seat in the game the observation belongs to.
        episodes: Games that finished during this call.
    """

    observations: Float[np.ndarray, "n obs_dim"]
    action_masks: Int8[np.ndarray, "n num_actions"]
    rewards: Float[np.ndarray, " n"]
    dones: Bool[np.ndarray, " n"]
    truncated: Bool[np.ndarray, " n"]
    learner_seats: Int[np.ndarray, " n"]
    episodes: list[EpisodeRecord] = field(default_factory=list[EpisodeRecord])
    final_observations: dict[int, np.ndarray] = field(default_factory=dict[int, np.ndarray])
    """For envs with ``truncated[i]``: the learner's observation of the cut-off position."""


class BatchedOpponentLoop:
    """Plays the opponent's turns so the learner sees fixed-shape batches.

    Args:
        batch: The environments (must be two-player).
        provider: Supplies opponents and receives finished-game results.
        actor_id: Passed to the provider (lets providers specialise per actor).
        seed: Seeds seat assignment, game seeds and opponent sampling.
    """

    def __init__(
        self, batch: AECBatch, provider: OpponentProvider, actor_id: int = 0, seed: int = 0
    ) -> None:
        """Create the loop. Call :meth:`reset` before :meth:`step`."""
        if batch.num_seats != _NUM_SEATS:
            raise ValueError("BatchedOpponentLoop supports two-player games only")
        self.batch: AECBatch = batch
        self.provider: OpponentProvider = provider
        self.actor_id: int = actor_id
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._opponent_rng: np.random.Generator = np.random.default_rng(self._rng.integers(2**63))
        n = batch.num_envs
        self._env_ids: np.ndarray = np.arange(n)
        self._learner_seat: np.ndarray = np.zeros(n, dtype=np.int64)
        self._to_move: np.ndarray = np.full(n, -1, dtype=np.int64)
        self._obs: np.ndarray = np.zeros((n, *batch.obs_shape), dtype=batch.obs_dtype)
        self._mask: np.ndarray = np.zeros((n, batch.num_actions), dtype=np.int8)
        self._info: list[str] = [""] * n
        self._env_opponent: list[OpponentId] = [""] * n
        self._episode_return: np.ndarray = np.zeros(n, dtype=np.float64)
        self._episode_decisions: np.ndarray = np.zeros(n, dtype=np.int64)
        # Learner rewards received in the current game before its first decision; they
        # are added to that game's first transition.
        self._carry: np.ndarray = np.zeros(n, dtype=np.float64)
        self._opponents: dict[OpponentId, Opponent] = {}
        self._current: tuple[OpponentId, Opponent] | None = None
        self._started: bool = False
        self.skipped_episodes: int = 0
        """Games that ended before the learner made a single decision."""

    # ------------------------------------------------------------------ opponents
    def set_opponent(self, opponent_id: OpponentId, opponent: Opponent) -> None:
        """Use this opponent for every game that starts from now on."""
        self._current = (opponent_id, opponent)
        self._opponents[opponent_id] = opponent

    def next_opponent(self) -> OpponentId:
        """Ask the provider for the next opponent and use it for new games."""
        opponent_id, opponent = self.provider.next_opponent(self.actor_id)
        self.set_opponent(opponent_id, opponent)
        return opponent_id

    # ------------------------------------------------------------------ API
    def reset(self) -> LearnerStep:
        """Start a game in every env and play up to the learner's first decision."""
        if self._current is None:
            self.next_opponent()
        acc = _Accumulator(self.batch.num_envs)
        self._settle(list(range(self.batch.num_envs)), acc)
        self._started = True
        return self._learner_step(acc)

    def step(self, actions: Int[np.ndarray, " n"]) -> LearnerStep:
        """Apply the learner's actions, play the opponents, and return the next batch."""
        if not self._started:
            raise RuntimeError("call reset() before step()")
        actions = np.asarray(actions, dtype=np.int64)
        n = self.batch.num_envs
        if actions.shape != (n,):
            raise ValueError(f"expected {n} actions, got shape {actions.shape}")
        legal = self._mask[self._env_ids, actions]
        if not legal.all():
            bad = np.flatnonzero(legal == 0).tolist()
            raise ValueError(f"illegal learner actions in envs {bad}")

        acc = _Accumulator(n)
        acc.rewards += self._carry  # this is the first decision in those games
        self._carry[:] = 0.0
        self._episode_decisions += 1
        to_reset = self._apply(self.batch.step(self._env_ids, actions), acc)
        self._settle(to_reset, acc)
        return self._learner_step(acc)

    # ------------------------------------------------------------------ internals
    def _settle(self, to_reset: list[int], acc: _Accumulator) -> None:
        """Reset finished envs and play opponent turns until the learner is up everywhere."""
        while True:
            if to_reset:
                to_reset = self._apply(self._reset_envs(np.asarray(to_reset)), acc)
                continue
            waiting = np.flatnonzero((self._to_move >= 0) & (self._to_move != self._learner_seat))
            if waiting.size == 0:
                return
            groups: defaultdict[OpponentId, list[int]] = defaultdict(list)
            for i in waiting:
                groups[self._env_opponent[i]].append(int(i))
            for opponent_id, members in groups.items():
                ids = np.asarray(members)
                obatch = OpponentBatch(
                    observations=self._obs[ids],
                    action_masks=self._mask[ids],
                    info_states=[self._info[i] for i in members],
                )
                actions = self._opponents[opponent_id].act(obatch, self._opponent_rng)
                to_reset += self._apply(
                    self.batch.step(ids, np.asarray(actions, dtype=np.int64)), acc
                )

    def _apply(self, view: BatchView, acc: _Accumulator) -> list[int]:
        """Absorb a view: store state, credit rewards, close finished games.

        Learner-seat rewards go to the current transition if the learner has already
        acted in this game. Otherwise they are held in ``_carry`` and added to the game's
        first transition on the next :meth:`step`.

        Returns:
            Ids of envs whose game finished and must be reset.
        """
        ids = view.env_ids
        seat_r = view.rewards[np.arange(ids.shape[0]), self._learner_seat[ids]]
        acted = self._episode_decisions[ids] > 0
        acc.rewards[ids] += np.where(acted, seat_r, 0.0)
        self._carry[ids] += np.where(acted, 0.0, seat_r)
        self._episode_return[ids] += seat_r
        self._to_move[ids] = view.to_move
        self._obs[ids] = view.observations
        self._mask[ids] = view.action_masks
        for row, i in enumerate(ids):
            self._info[i] = view.info_states[row]

        finished: list[int] = []
        for row in np.flatnonzero(view.done):
            i = int(ids[row])
            truncated = bool(view.truncated[row])
            if self._episode_decisions[i] > 0:
                acc.dones[i] = True
                acc.truncated[i] = truncated
                final = view.final_observations.get(int(row)) if truncated else None
                if final is not None:
                    acc.final_obs[i] = final[self._learner_seat[i]]
            else:
                self.skipped_episodes += 1
                self._carry[i] = 0.0  # no learner transition to credit
            acc.episodes.append(self._finish(i, truncated=truncated))
            finished.append(i)
        return finished

    def _finish(self, i: int, *, truncated: bool) -> EpisodeRecord:
        record = EpisodeRecord(
            env_id=i,
            opponent_id=self._env_opponent[i],
            learner_seat=int(self._learner_seat[i]),
            learner_return=float(self._episode_return[i]),
            learner_decisions=int(self._episode_decisions[i]),
            truncated=truncated,
        )
        self.provider.report(record.opponent_id, record.learner_return)
        return record

    def _reset_envs(self, ids: np.ndarray) -> BatchView:
        """Start new games in ``ids`` with fresh seats, seeds and the current opponent."""
        assert self._current is not None
        opponent_id = self._current[0]
        self._learner_seat[ids] = self._rng.integers(0, _NUM_SEATS, size=ids.shape[0])
        for i in ids:
            self._env_opponent[i] = opponent_id
        self._episode_return[ids] = 0.0
        self._episode_decisions[ids] = 0
        self._carry[ids] = 0.0
        seeds = self._rng.integers(0, 2**31 - 1, size=ids.shape[0])
        return self.batch.reset(ids, seeds)

    def _learner_step(self, acc: _Accumulator) -> LearnerStep:
        return LearnerStep(
            observations=self._obs.copy(),
            action_masks=self._mask.copy(),
            rewards=acc.rewards,
            dones=acc.dones,
            truncated=acc.truncated,
            learner_seats=self._learner_seat.copy(),
            episodes=acc.episodes,
            final_observations=acc.final_obs,
        )


class _Accumulator:
    """Per-call scratch space for building a :class:`LearnerStep`."""

    def __init__(self, n: int) -> None:
        self.rewards: np.ndarray = np.zeros(n, dtype=np.float64)
        self.dones: np.ndarray = np.zeros(n, dtype=bool)
        self.truncated: np.ndarray = np.zeros(n, dtype=bool)
        self.episodes: list[EpisodeRecord] = []
        self.final_obs: dict[int, np.ndarray] = {}
