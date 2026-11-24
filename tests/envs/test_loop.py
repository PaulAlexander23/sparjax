from collections import defaultdict

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sparjax.envs.batch import InProcessAECBatch
from sparjax.envs.loop import BatchedOpponentLoop, EpisodeRecord, LearnerStep
from sparjax.envs.openspiel_aec import OpenSpielAECEnv
from sparjax.opponents import FixedProvider, RandomOpponent, ScriptedOpponent
from tests.envs.fakes import ScriptedGame


def random_actions(step: LearnerStep, rng: np.random.Generator) -> np.ndarray:
    m = step.action_masks.astype(float)
    return np.array([rng.choice(np.flatnonzero(row)) for row in m])


def run_and_check(loop: BatchedOpponentLoop, steps: int, seed: int = 0) -> list[EpisodeRecord]:
    """Drive the loop with a random learner and assert the core invariants on every step."""
    rng = np.random.default_rng(seed)
    batch = loop.batch
    assert isinstance(batch, InProcessAECBatch)
    n = batch.num_envs
    step = loop.reset()
    running = np.zeros(n)
    all_episodes: list[EpisodeRecord] = []
    for _ in range(steps):
        # Learner is never asked to act off-turn.
        for i, env in enumerate(batch.envs):
            assert env.agent_selection == env.possible_agents[step.learner_seats[i]]
        assert step.action_masks.any(axis=1).all()
        step = loop.step(random_actions(step, rng))
        # Shapes are constant.
        assert step.observations.shape == (n, *batch.obs_shape)
        assert step.rewards.shape == step.dones.shape == (n,)
        running += step.rewards
        finished: defaultdict[int, list[EpisodeRecord]] = defaultdict(list)
        for ep in step.episodes:
            finished[ep.env_id].append(ep)
        for i in np.flatnonzero(step.dones):
            credited = [ep for ep in finished[int(i)] if ep.learner_decisions > 0]
            assert len(credited) == 1
            # Sum of per-step learner rewards over a game equals that game's return.
            assert running[i] == pytest.approx(credited[0].learner_return)
            running[i] = 0.0
        all_episodes += step.episodes
    return all_episodes


@pytest.mark.parametrize("game", ["kuhn_poker", "leduc_poker"])
def test_random_vs_random_poker_10k_steps(game: str) -> None:
    batch = InProcessAECBatch(lambda: OpenSpielAECEnv(game), 16)
    provider = FixedProvider(RandomOpponent(), "random")
    loop = BatchedOpponentLoop(batch, provider, seed=1)
    episodes = run_and_check(loop, steps=10_000 // 16)
    assert loop.skipped_episodes == 0
    returns = np.array([ep.learner_return for ep in episodes])
    if game == "kuhn_poker":
        assert np.all(returns != 0)  # Kuhn has no ties (Leduc can split the pot)
    seats = np.array([ep.learner_seat for ep in episodes])
    assert abs(seats.mean() - 0.5) < 0.05  # seats balanced
    assert provider.stats()["random"].games == len(episodes)


@given(
    length=st.integers(1, 7),
    first=st.integers(0, 1),
    n=st.integers(1, 6),
    reset_reward=st.sampled_from([0.0, 0.5]),
    seed=st.integers(0, 1000),
)
@settings(max_examples=60, deadline=None)
def test_reward_accounting_property(
    length: int, first: int, n: int, reset_reward: float, seed: int
) -> None:
    batch = InProcessAECBatch(
        lambda: ScriptedGame(length=length, first=first, reset_reward=reset_reward), n
    )
    loop = BatchedOpponentLoop(batch, FixedProvider(RandomOpponent()), seed=seed)
    episodes = run_and_check(loop, steps=25, seed=seed)
    for ep in episodes:
        assert ep.learner_decisions <= (length + 1) // 2 + 1


def test_rewards_before_first_decision_are_carried() -> None:
    # Seat 0 always opens with action 2, so a learner in seat 1 gets -3 before acting.
    always_2 = ScriptedOpponent(lambda b: np.tile([0.0, 0.0, 1.0], (b.size, 1)))
    for seed in range(100):
        batch = InProcessAECBatch(lambda: ScriptedGame(length=4, first=0), 1)
        loop = BatchedOpponentLoop(batch, FixedProvider(always_2), seed=seed)
        step = loop.reset()
        if step.learner_seats[0] == 1:
            break
    else:
        pytest.fail("no seed put the learner in seat 1")
    assert step.rewards.tolist() == [0.0]  # reset() never returns reward
    step = loop.step(np.array([0]))  # learner +1, then the opponent's 2 costs it 3
    # The first transition also carries the -3 received before the first decision.
    assert step.rewards.tolist() == [-5.0]


def test_skipped_episodes_when_game_ends_before_learner_acts() -> None:
    batch = InProcessAECBatch(lambda: ScriptedGame(length=1, first=0), 4)
    loop = BatchedOpponentLoop(batch, FixedProvider(RandomOpponent()), seed=3)
    run_and_check(loop, steps=20)
    assert loop.skipped_episodes > 0


def test_opponent_switch_only_affects_new_games() -> None:
    batch = InProcessAECBatch(lambda: ScriptedGame(length=6), 8)
    loop = BatchedOpponentLoop(batch, FixedProvider(RandomOpponent(), "a"), seed=0)
    rng = np.random.default_rng(0)
    step = loop.reset()
    step = loop.step(random_actions(step, rng))
    loop.set_opponent("b", RandomOpponent())
    in_progress = {i for i in range(8)} - {ep.env_id for ep in step.episodes}
    seen_first: dict[int, str] = {}
    all_ids: list[str] = []
    for _ in range(10):
        step = loop.step(random_actions(step, rng))
        for ep in step.episodes:
            seen_first.setdefault(ep.env_id, ep.opponent_id)
        all_ids.extend(ep.opponent_id for ep in step.episodes)
    for i in in_progress:
        assert seen_first[i] == "a"
    assert "b" in all_ids


def test_next_opponent_uses_provider() -> None:
    batch = InProcessAECBatch(ScriptedGame, 1)
    loop = BatchedOpponentLoop(batch, FixedProvider(RandomOpponent(), "rnd"), actor_id=3)
    assert loop.next_opponent() == "rnd"


def test_input_validation() -> None:
    loop = BatchedOpponentLoop(InProcessAECBatch(ScriptedGame, 2), FixedProvider(RandomOpponent()))
    with pytest.raises(RuntimeError, match="reset"):
        loop.step(np.zeros(2, dtype=np.int64))
    loop.reset()
    with pytest.raises(ValueError, match="expected 2 actions"):
        loop.step(np.zeros(3, dtype=np.int64))
    loop2 = BatchedOpponentLoop(
        InProcessAECBatch(lambda: ScriptedGame(legal=(1,)), 1), FixedProvider(RandomOpponent())
    )
    loop2.reset()
    with pytest.raises(ValueError, match="illegal"):
        loop2.step(np.array([0]))
    with pytest.raises(ValueError, match="two-player"):
        BatchedOpponentLoop(
            InProcessAECBatch(lambda: ScriptedGame(num_agents_=3), 1),
            FixedProvider(RandomOpponent()),
        )


def test_truncation_propagates() -> None:
    batch = InProcessAECBatch(lambda: ScriptedGame(length=2, truncate=True), 2)
    loop = BatchedOpponentLoop(batch, FixedProvider(RandomOpponent()), seed=0)
    rng = np.random.default_rng(0)
    step = loop.reset()
    saw = False
    for _ in range(10):
        step = loop.step(random_actions(step, rng))
        if step.dones.any():
            assert (step.truncated == step.dones).all()
            saw = True
    assert saw
