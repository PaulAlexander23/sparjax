import numpy as np
import pytest

from sparjax.envs.batch import InProcessAECBatch
from sparjax.envs.loop import BatchedOpponentLoop
from sparjax.envs.registry import EnvSpec
from sparjax.opponents import FixedProvider, OpponentBatch, RandomOpponent
from sparjax.opponents.go import GoLightPlayoutOpponent, own_eye_points


def test_own_eye_points() -> None:
    own = np.zeros((1, 3, 3), bool)
    opp = np.zeros((1, 3, 3), bool)
    own[0, 0, 1] = own[0, 1, 0] = True  # corner (0, 0) is an eye
    own[0, 1, 2] = own[0, 2, 1] = True
    opp[0, 1, 1] = True  # centre occupied
    eyes = own_eye_points(own, opp)
    assert eyes[0, 0, 0]
    assert eyes[0, 2, 2]
    assert not eyes[0, 1, 1]
    assert eyes[0, 0, 2]  # corners (0, 2) and (2, 0) are also enclosed by own stones
    assert eyes[0, 2, 0]
    assert eyes.sum() == 4


def test_never_fills_own_eye_or_passes_when_moves_exist() -> None:
    n = 3
    obs = np.zeros((1, n, n, 17), bool)
    obs[0, 0, 1, 1] = obs[0, 1, 0, 1] = True  # own stones around corner (0, 0)
    mask = np.zeros((1, n * n + 1), np.int8)
    mask[0, 0] = 1  # the eye
    mask[0, 8] = 1  # a normal point
    mask[0, n * n] = 1  # pass
    acts = GoLightPlayoutOpponent().act(OpponentBatch(obs, mask), np.random.default_rng(0))
    assert acts.tolist() == [8]
    only_eye = mask.copy()
    only_eye[0, 8] = 0
    act = GoLightPlayoutOpponent().act(OpponentBatch(obs, only_eye), np.random.default_rng(0))
    assert act[0] in (0, n * n)  # falls back to any legal move


def test_rejects_non_go_observations() -> None:
    with pytest.raises(ValueError, match="Go observations"):
        GoLightPlayoutOpponent().act(
            OpponentBatch(np.zeros((1, 5), np.float32), np.ones((1, 3), np.int8)),
            np.random.default_rng(0),
        )


def test_beats_random_on_small_board() -> None:
    spec = EnvSpec("pettingzoo:classic/go-v5", {"board_size": 5})
    loop = BatchedOpponentLoop(InProcessAECBatch(spec, 16), FixedProvider(RandomOpponent()), seed=0)
    me, rng = GoLightPlayoutOpponent(), np.random.default_rng(0)
    step, results = loop.reset(), []
    while len(results) < 100:
        step = loop.step(me.act(OpponentBatch(step.observations, step.action_masks), rng))
        results += [ep.learner_return for ep in step.episodes]
    assert np.mean(np.array(results) > 0) > 0.6
