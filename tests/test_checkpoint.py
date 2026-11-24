from pathlib import Path

import numpy as np
import pytest

from sparjax.agents.ppo.networks import init_actor_critic
from sparjax.checkpoint import (
    _flatten,
    _unflatten,
    config_to_dict,
    load_checkpoint,
    save_checkpoint,
)
from sparjax.config import TrainConfig


def test_round_trip(tmp_path: Path) -> None:
    graphdef, params = init_actor_critic(5, 3, (8, 4), seed=1)
    path = save_checkpoint(
        tmp_path / "c" / "x.npz", params, obs_shape=5, num_actions=3, hidden=(8, 4), meta={"a": 1}
    )
    ckpt = load_checkpoint(path)
    assert ckpt.meta == {"a": 1}
    from flax import nnx

    a, b = nnx.to_pure_dict(params), nnx.to_pure_dict(ckpt.params)
    fa, fb = _flatten(a), _flatten(b)
    assert fa.keys() == fb.keys()
    for k in fa:
        np.testing.assert_array_equal(fa[k], fb[k])
    del graphdef


def test_flatten_unflatten_inverse() -> None:
    tree = {"a": {0: np.ones(2), 1: {"w": np.zeros(1)}}, "b": np.arange(3)}
    back = _unflatten(_flatten(tree))
    assert set(back) == {"a", "b"}
    np.testing.assert_array_equal(back["a"][1]["w"], np.zeros(1))


def test_rejects_foreign_npz(tmp_path: Path) -> None:
    p = tmp_path / "f.npz"
    np.savez(p, __header__=np.array('{"format": "nope"}'))
    with pytest.raises(ValueError, match="not a sparjax checkpoint"):
        load_checkpoint(p)


def test_config_to_dict_is_json_friendly() -> None:
    d = config_to_dict(TrainConfig())
    assert d["env"]["id"] == "openspiel:kuhn_poker"
    assert d["log_dir"] is None
