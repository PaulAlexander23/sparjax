import threading
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sparjax.opponents import (
    FixedProvider,
    MixtureEntry,
    MixtureProvider,
    OpponentBatch,
    OutcomeStats,
    RandomOpponent,
    ScriptedOpponent,
    TabularPolicy,
    TabularPolicyOpponent,
    sample_actions,
)


def make_batch(masks: list[list[int]], info_states: list[str] | None = None) -> OpponentBatch:
    m = np.asarray(masks, dtype=np.int8)
    return OpponentBatch(np.zeros((m.shape[0], 3), np.float32), m, info_states)


# ---------------------------------------------------------------- sample_actions
@st.composite
def probs_and_masks(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    b = draw(st.integers(1, 16))
    a = draw(st.integers(1, 6))
    masks = draw(arrays(np.int8, (b, a), elements=st.integers(0, 1)))
    masks[np.arange(b), draw(arrays(np.int64, b, elements=st.integers(0, a - 1)))] = 1
    probs = draw(arrays(np.float64, (b, a), elements=st.floats(0, 10, allow_subnormal=False)))
    return probs, masks


@given(probs_and_masks(), st.integers(0, 2**32 - 1))
@settings(max_examples=200, deadline=None)
def test_sample_actions_always_legal(pm: tuple[np.ndarray, np.ndarray], seed: int) -> None:
    probs, masks = pm
    actions = sample_actions(probs, masks, np.random.default_rng(seed))
    assert actions.shape == (masks.shape[0],)
    assert masks[np.arange(masks.shape[0]), actions].all()


def test_sample_actions_matches_distribution() -> None:
    probs = np.tile([0.2, 0.0, 0.8], (20000, 1))
    masks = np.ones_like(probs, dtype=np.int8)
    actions = sample_actions(probs, masks, np.random.default_rng(0))
    freqs = np.bincount(actions, minlength=3) / len(actions)
    np.testing.assert_allclose(freqs, [0.2, 0.0, 0.8], atol=0.01)


def test_sample_actions_zero_mass_on_legal_falls_back_to_uniform() -> None:
    probs = np.tile([1.0, 0.0, 0.0], (10000, 1))
    masks = np.tile(np.array([0, 1, 1], np.int8), (10000, 1))
    actions = sample_actions(probs, masks, np.random.default_rng(0))
    freqs = np.bincount(actions, minlength=3) / len(actions)
    np.testing.assert_allclose(freqs, [0.0, 0.5, 0.5], atol=0.02)


# ---------------------------------------------------------------- OpponentBatch
def test_batch_validation() -> None:
    with pytest.raises(ValueError, match="batch sizes"):
        OpponentBatch(np.zeros((2, 3), np.float32), np.ones((3, 2), np.int8))
    with pytest.raises(ValueError, match="info_states"):
        make_batch([[1, 1]], ["a", "b"])
    with pytest.raises(ValueError, match="at least one legal"):
        make_batch([[0, 0]])
    assert make_batch([[1, 0], [0, 1]]).size == 2


# ---------------------------------------------------------------- simple opponents
def test_random_opponent_uniform_over_legal() -> None:
    batch = make_batch([[1, 0, 1]] * 10000)
    actions = RandomOpponent().act(batch, np.random.default_rng(0))
    freqs = np.bincount(actions, minlength=3) / len(actions)
    np.testing.assert_allclose(freqs, [0.5, 0.0, 0.5], atol=0.02)


def test_scripted_opponent_prefers_but_respects_mask() -> None:
    opp = ScriptedOpponent(lambda b: np.tile([0.0, 1.0, 0.0], (b.size, 1)))
    actions = opp.act(make_batch([[1, 1, 1], [1, 0, 1]]), np.random.default_rng(0))
    assert actions[0] == 1
    assert actions[1] in (0, 2)


def test_scripted_opponent_validates_output() -> None:
    with pytest.raises(ValueError, match="shape"):
        ScriptedOpponent(lambda b: np.ones((b.size, 5))).act(
            make_batch([[1, 1]]), np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="negative"):
        ScriptedOpponent(lambda b: -np.ones((b.size, 2))).act(
            make_batch([[1, 1]]), np.random.default_rng(0)
        )


# ---------------------------------------------------------------- tabular
@pytest.fixture
def policy() -> TabularPolicy:
    return TabularPolicy(
        game_name="kuhn_poker",
        num_actions=2,
        probs={"0": np.array([1.0, 0.0]), "1b": np.array([0.25, 0.75])},
        metadata={"solver": "test", "iterations": 3},
    )


def test_tabular_round_trip(policy: TabularPolicy, tmp_path: Path) -> None:
    path = policy.save(tmp_path / "p.npz")
    loaded = TabularPolicy.load(path)
    assert loaded.game_name == policy.game_name
    assert loaded.num_actions == policy.num_actions
    assert dict(loaded.metadata) == dict(policy.metadata)
    assert loaded.probs.keys() == policy.probs.keys()
    for k in policy.probs:
        np.testing.assert_array_equal(loaded.probs[k], policy.probs[k])
    assert len(loaded) == 2


def test_tabular_round_trip_empty(tmp_path: Path) -> None:
    empty = TabularPolicy("kuhn_poker", 2, {})
    assert len(TabularPolicy.load(empty.save(tmp_path / "e.npz"))) == 0


def test_tabular_load_rejects_other_npz(tmp_path: Path) -> None:
    path = tmp_path / "x.npz"
    np.savez(path, header=np.array('{"format": "other"}'))
    with pytest.raises(ValueError, match="not a sparjax"):
        TabularPolicy.load(path)


def test_tabular_validation() -> None:
    with pytest.raises(ValueError, match="shape"):
        TabularPolicy("g", 2, {"a": np.array([1.0])})
    with pytest.raises(ValueError, match="probability"):
        TabularPolicy("g", 2, {"a": np.array([0.7, 0.7])})


def test_tabular_opponent_follows_table(policy: TabularPolicy) -> None:
    opp = TabularPolicyOpponent(policy)
    n = 20000
    batch = make_batch([[1, 1]] * n, ["1b"] * n)
    freqs = np.bincount(opp.act(batch, np.random.default_rng(0)), minlength=2) / n
    np.testing.assert_allclose(freqs, [0.25, 0.75], atol=0.01)
    assert opp.act(make_batch([[1, 1]], ["0"]), np.random.default_rng(0))[0] == 0


def test_tabular_opponent_missing_state(policy: TabularPolicy) -> None:
    batch = make_batch([[1, 1]], ["unknown"])
    with pytest.raises(KeyError, match="unknown"):
        TabularPolicyOpponent(policy).act(batch, np.random.default_rng(0))
    action = TabularPolicyOpponent(policy, on_missing="uniform").act(
        batch, np.random.default_rng(0)
    )
    assert action[0] in (0, 1)


def test_tabular_opponent_input_checks(policy: TabularPolicy) -> None:
    opp = TabularPolicyOpponent(policy)
    with pytest.raises(ValueError, match="info_states"):
        opp.act(make_batch([[1, 1]]), np.random.default_rng(0))
    with pytest.raises(ValueError, match="action space"):
        opp.act(make_batch([[1, 1, 1]], ["0"]), np.random.default_rng(0))


# ---------------------------------------------------------------- providers
def test_outcome_stats() -> None:
    s = OutcomeStats()
    assert s.mean == 0.0
    assert s.stderr == float("inf")
    for x in [1.0, -1.0, 1.0, 3.0]:
        s.add(x)
    assert s.games == 4
    assert s.mean == pytest.approx(1.0)
    assert s.stderr == pytest.approx(np.std([1, -1, 1, 3], ddof=1) / 2)


def test_fixed_provider_and_thread_safe_reporting() -> None:
    opp = RandomOpponent()
    provider = FixedProvider(opp, "rand")
    assert provider.next_opponent(0) == ("rand", opp)

    def worker() -> None:
        for _ in range(1000):
            provider.report("rand", 1.0)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stats = provider.stats()["rand"]
    assert stats.games == 8000
    assert stats.mean == pytest.approx(1.0)


def test_mixture_provider_frequencies() -> None:
    a, b = RandomOpponent(), RandomOpponent()
    provider = MixtureProvider([MixtureEntry("a", a, 3.0), MixtureEntry("b", b, 1.0)], seed=0)
    ids = [provider.next_opponent(i)[0] for i in range(8000)]
    assert ids.count("a") / len(ids) == pytest.approx(0.75, abs=0.02)


@pytest.mark.parametrize(
    ("entries", "match"),
    [
        ([], "at least one"),
        ([MixtureEntry("a", RandomOpponent(), 1.0)] * 2, "unique"),
        ([MixtureEntry("a", RandomOpponent(), 0.0)], "positive"),
    ],
)
def test_mixture_provider_validation(entries: list[MixtureEntry], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MixtureProvider(entries)
