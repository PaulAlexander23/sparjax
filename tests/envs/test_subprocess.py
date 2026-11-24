import numpy as np
import pytest

from sparjax.envs.batch import BatchView, InProcessAECBatch
from sparjax.envs.loop import BatchedOpponentLoop
from sparjax.envs.registry import EnvSpec
from sparjax.envs.subprocess import SubprocessAECBatch
from sparjax.opponents import FixedProvider, RandomOpponent

KUHN = EnvSpec("openspiel:kuhn_poker")
GO = EnvSpec("pettingzoo:classic/go-v5", {"board_size": 5})


def assert_views_equal(a: BatchView, b: BatchView) -> None:
    np.testing.assert_array_equal(a.env_ids, b.env_ids)
    np.testing.assert_array_equal(a.to_move, b.to_move)
    np.testing.assert_array_equal(a.observations, b.observations)
    np.testing.assert_array_equal(a.action_masks, b.action_masks)
    np.testing.assert_array_equal(a.rewards, b.rewards)
    np.testing.assert_array_equal(a.done, b.done)
    np.testing.assert_array_equal(a.truncated, b.truncated)
    assert a.info_states == b.info_states


@pytest.mark.parametrize("spec", [KUHN, GO], ids=["kuhn", "go5"])
def test_subprocess_matches_in_process(spec: EnvSpec) -> None:
    n = 7
    local = InProcessAECBatch(spec, n)
    remote = SubprocessAECBatch(spec, n, num_workers=3)
    try:
        assert (remote.num_envs, remote.num_workers) == (n, 3)
        assert (remote.num_seats, remote.obs_shape, remote.num_actions) == (
            local.num_seats,
            local.obs_shape,
            local.num_actions,
        )
        rng = np.random.default_rng(0)
        ids = np.array([6, 0, 3, 1, 5, 2, 4])  # deliberately out of order
        seeds = np.arange(n) + 11
        view_l, view_r = local.reset(ids, seeds), remote.reset(ids, seeds)
        assert_views_equal(view_l, view_r)
        for _ in range(40):
            live = view_l.env_ids[~view_l.done]
            done = view_l.env_ids[view_l.done]
            if done.size:
                seeds = rng.integers(0, 1000, size=done.size)
                assert_views_equal(local.reset(done, seeds), remote.reset(done, seeds))
            if live.size == 0:
                view_l = local.reset(ids, seeds := rng.integers(0, 1000, size=n))
                view_r = remote.reset(ids, seeds)
                continue
            masks = view_l.action_masks[~view_l.done]
            actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
            subset = rng.permutation(live.size)[: max(1, live.size // 2)]
            view_l = local.step(live[subset], actions[subset])
            view_r = remote.step(live[subset], actions[subset])
            assert_views_equal(view_l, view_r)
    finally:
        remote.close()
        local.close()


def test_loop_trajectories_identical_over_subprocesses() -> None:
    def run(batch: InProcessAECBatch | SubprocessAECBatch) -> list[np.ndarray]:
        loop = BatchedOpponentLoop(batch, FixedProvider(RandomOpponent()), seed=5)
        rng = np.random.default_rng(1)
        step = loop.reset()
        out = []
        for _ in range(30):
            a = np.array([rng.choice(np.flatnonzero(m)) for m in step.action_masks])
            step = loop.step(a)
            out += [step.observations, step.rewards, step.dones.astype(float)]
        return out

    remote = SubprocessAECBatch(GO, 6, num_workers=2)
    try:
        for x, y in zip(run(InProcessAECBatch(GO, 6)), run(remote), strict=True):
            np.testing.assert_array_equal(x, y)
    finally:
        remote.close()


def test_worker_errors_are_raised() -> None:
    remote = SubprocessAECBatch(KUHN, 2, num_workers=2)
    try:
        remote.reset(np.array([0, 1]), np.array([0, 1]))
        with pytest.raises(RuntimeError, match="illegal action"):
            remote.step(np.array([0]), np.array([7]))
        with pytest.raises(ValueError, match="same shape"):
            remote.step(np.array([0, 1]), np.array([0]))
    finally:
        remote.close()
    remote.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        remote.step(np.array([0]), np.array([0]))


def test_bad_env_spec_fails_at_startup() -> None:
    with pytest.raises(RuntimeError, match="failed"):
        SubprocessAECBatch(EnvSpec("openspiel:kuhn_poker", {"players": 3}), 1, num_workers=1)


def test_argument_validation() -> None:
    with pytest.raises(ValueError, match="num_envs"):
        SubprocessAECBatch(KUHN, 0, num_workers=1)
    with pytest.raises(ValueError, match="num_workers"):
        SubprocessAECBatch(KUHN, 2, num_workers=3)


def test_env_spec_validation_and_properties() -> None:
    assert KUHN.backend == "openspiel"
    assert KUHN.openspiel_game == "kuhn_poker"
    assert GO.openspiel_game is None
    assert GO.name == "classic/go-v5"
    with pytest.raises(ValueError, match="env id"):
        EnvSpec("gym:CartPole")
    with pytest.raises(ValueError, match="env id"):
        EnvSpec("openspiel:")
