from typing import Literal

import numpy as np
import pytest

from sparjax.envs.batch import InProcessAECBatch
from sparjax.envs.openspiel_aec import OpenSpielAECEnv
from tests.envs.fakes import ScriptedGame


def ids(*xs: int) -> np.ndarray:
    return np.asarray(xs, dtype=np.int64)


@pytest.mark.parametrize("mask_mode", ["obs", "info", "none"])
def test_reset_and_step_report_turn_obs_and_mask(mask_mode: Literal["obs", "info", "none"]) -> None:
    batch = InProcessAECBatch(lambda: ScriptedGame(mask_mode=mask_mode, legal=(0, 2)), 2)
    assert (batch.num_envs, batch.num_seats, batch.obs_shape, batch.num_actions) == (2, 2, (2,), 3)
    view = batch.reset(ids(0, 1), ids(5, 6))
    assert view.to_move.tolist() == [0, 0]
    np.testing.assert_array_equal(view.observations[0], [0, 0])
    expected_mask = [1, 1, 1] if mask_mode == "none" else [1, 0, 1]
    assert view.action_masks[0].tolist() == expected_mask
    assert view.info_states == ["p0:0", "p0:0"]
    assert [env.seeds for env in batch.envs] == [[5], [6]]  # pyright: ignore[reportAttributeAccessIssue]

    view = batch.step(ids(1), ids(2))
    assert view.env_ids.tolist() == [1]
    assert view.to_move.tolist() == [1]
    assert view.rewards.tolist() == [[3.0, -3.0]]
    assert not view.done.any()


def test_game_end_drains_agents_and_sets_flags() -> None:
    batch = InProcessAECBatch(lambda: ScriptedGame(length=2), 1)
    batch.reset(ids(0), ids(0))
    batch.step(ids(0), ids(0))
    view = batch.step(ids(0), ids(1))
    assert view.done.tolist() == [True]
    assert view.truncated.tolist() == [False]
    assert view.to_move.tolist() == [-1]
    assert view.rewards.tolist() == [[-2.0, 2.0]]
    assert not view.action_masks.any()
    assert batch.envs[0].agents == []
    with pytest.raises(RuntimeError, match="reset it"):
        batch.step(ids(0), ids(0))


def test_truncation_flag() -> None:
    batch = InProcessAECBatch(lambda: ScriptedGame(length=1, truncate=True), 1)
    batch.reset(ids(0), ids(0))
    view = batch.step(ids(0), ids(0))
    assert view.done.tolist() == [True]
    assert view.truncated.tolist() == [True]


def test_game_over_at_reset_and_reset_rewards() -> None:
    batch = InProcessAECBatch(lambda: ScriptedGame(length=0, reset_reward=0.5), 1)
    view = batch.reset(ids(0), ids(0))
    assert view.done.tolist() == [True]
    assert view.rewards.tolist() == [[0.5, -0.5]]


def test_argument_validation() -> None:
    with pytest.raises(ValueError, match="num_envs"):
        InProcessAECBatch(ScriptedGame, 0)
    batch = InProcessAECBatch(ScriptedGame, 2)
    with pytest.raises(ValueError, match="same shape"):
        batch.reset(ids(0, 1), ids(0))
    batch.reset(ids(0, 1), ids(0, 1))
    with pytest.raises(ValueError, match="same shape"):
        batch.step(ids(0, 1), ids(0))
    batch.close()


def test_openspiel_batch_matches_single_env() -> None:
    batch = InProcessAECBatch(lambda: OpenSpielAECEnv("leduc_poker"), 3)
    assert batch.obs_shape == (30,)
    view = batch.reset(ids(0, 1, 2), ids(1, 2, 3))
    single = OpenSpielAECEnv("leduc_poker")
    single.reset(seed=2)
    obs = single.observe(single.agent_selection)
    np.testing.assert_array_equal(view.observations[1], obs["observation"])
    np.testing.assert_array_equal(view.action_masks[1], obs["action_mask"])
