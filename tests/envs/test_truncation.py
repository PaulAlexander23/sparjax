import numpy as np
import pytest

from sparjax.envs.batch import InProcessAECBatch
from sparjax.envs.loop import BatchedOpponentLoop
from sparjax.envs.registry import EnvSpec
from sparjax.envs.subprocess import SubprocessAECBatch
from sparjax.envs.wrappers import MaxStepsWrapper
from sparjax.opponents import FixedProvider, RandomOpponent
from tests.envs.fakes import ScriptedGame

GO5 = EnvSpec("pettingzoo:classic/go-v5", {"board_size": 5}, max_steps=6)


def test_max_steps_wrapper_truncates_with_zero_reward() -> None:
    env = GO5()
    assert isinstance(env, MaxStepsWrapper)
    env.reset(seed=0)
    moves = 0
    for _ in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        if term or trunc:
            assert trunc
            assert not term
            assert reward == 0.0
            env.step(None)
            continue
        assert obs is not None
        env.step(int(np.flatnonzero(obs["action_mask"])[0]))
        moves += 1
    assert moves == 6
    env.reset(seed=1)  # the counter resets
    assert env.agents


def test_max_steps_wrapper_keeps_natural_endings_and_validates() -> None:
    env = MaxStepsWrapper(ScriptedGame(length=2), max_steps=2)  # ends naturally on step 2
    env.reset(seed=0)
    env.step(0)
    env.step(0)
    assert all(env.terminations.values())
    assert not any(env.truncations.values())
    with pytest.raises(ValueError, match="max_steps"):
        MaxStepsWrapper(ScriptedGame(), max_steps=0)


@pytest.mark.parametrize("workers", [0, 2])
def test_batch_reports_final_observations_for_truncated_games(workers: int) -> None:
    batch = (
        InProcessAECBatch(lambda: ScriptedGame(length=2, truncate=True), 3)
        if workers == 0
        else SubprocessAECBatch(
            EnvSpec("pettingzoo:classic/go-v5", {"board_size": 5}, max_steps=2), 3, workers
        )
    )
    try:
        ids = np.arange(3)
        view = batch.reset(ids, ids)
        for _ in range(2):  # always play the first legal move
            view = batch.step(ids, view.action_masks.argmax(axis=1))
        assert view.done.all()
        assert view.truncated.all()
        assert sorted(view.final_observations) == [0, 1, 2]
        assert view.final_observations[0].shape == (2, *batch.obs_shape)
    finally:
        batch.close()


def test_terminated_games_have_no_final_observations() -> None:
    batch = InProcessAECBatch(lambda: ScriptedGame(length=1), 2)
    batch.reset(np.arange(2), np.arange(2))
    view = batch.step(np.arange(2), np.zeros(2, np.int64))
    assert view.done.all()
    assert not view.truncated.any()
    assert view.final_observations == {}


def test_loop_passes_learner_seat_final_observation() -> None:
    # ScriptedGame observations are [moves_so_far, observer_seat], so the final observation
    # handed to the learner must carry the learner's own seat.
    batch = InProcessAECBatch(lambda: ScriptedGame(length=3, truncate=True), 6)
    loop = BatchedOpponentLoop(batch, FixedProvider(RandomOpponent()), seed=0)
    rng = np.random.default_rng(0)
    step = loop.reset()
    seen = 0
    for _ in range(10):
        seats = step.learner_seats.copy()
        a = np.array([rng.choice(np.flatnonzero(m)) for m in step.action_masks])
        step = loop.step(a)
        assert set(step.final_observations) == set(np.flatnonzero(step.truncated).tolist())
        for i, final in step.final_observations.items():
            assert final[1] == seats[i]
            assert final[0] == 3  # the position after the last move
            seen += 1
    assert seen > 0
