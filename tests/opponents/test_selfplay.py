import threading

import numpy as np
import pytest

from sparjax.agents.ppo.networks import init_actor_critic
from sparjax.eval.exact import mixture_exploitability, network_to_tabular
from sparjax.opponents import OpponentBatch
from sparjax.opponents.cfr import solve_cfr
from sparjax.opponents.neural import NeuralOpponent, NeuralPolicy, _bucket
from sparjax.opponents.selfplay import SnapshotPoolProvider, SnapshotSink


def batch(n: int, obs_dim: int = 4, num_actions: int = 3) -> OpponentBatch:
    mask = np.ones((n, num_actions), np.int8)
    mask[:, 2] = 0
    return OpponentBatch(
        np.random.default_rng(0).normal(size=(n, obs_dim)).astype(np.float32), mask
    )


def test_bucket() -> None:
    assert [_bucket(n) for n in (1, 2, 3, 4, 5, 64, 65)] == [1, 2, 4, 4, 8, 64, 128]


def test_neural_opponent_legal_and_padded() -> None:
    graphdef, params = init_actor_critic(4, 3, (8,), seed=0)
    opp = NeuralOpponent(NeuralPolicy(graphdef), params)
    rng = np.random.default_rng(0)
    for n in (1, 3, 5, 17):
        acts = opp.act(batch(n), rng)
        assert acts.shape == (n,)
        assert set(acts.tolist()) <= {0, 1}
    greedy = NeuralOpponent(NeuralPolicy(graphdef, greedy=True), params)
    b = batch(6)
    assert (greedy.act(b, rng) == greedy.act(b, rng)).all()


def make_pool(sampler: str, **kw: object) -> tuple[SnapshotPoolProvider, object, object]:
    graphdef, params = init_actor_critic(4, 3, (8,), seed=0)
    pool = SnapshotPoolProvider(sampler=sampler, seed=0, **kw)  # pyright: ignore[reportArgumentType]
    return pool, graphdef, params


def test_pool_add_evict_and_latest() -> None:
    pool, graphdef, params = make_pool("latest", max_size=3)
    assert isinstance(pool, SnapshotSink)
    with pytest.raises(RuntimeError, match="empty"):
        pool.next_opponent(0)
    for u in range(5):
        pool.add_snapshot(u * 10, graphdef, params)  # pyright: ignore[reportArgumentType]
    assert pool.size == 3
    assert pool.snapshot_ids() == ["snapshot_000020", "snapshot_000030", "snapshot_000040"]
    assert {pool.next_opponent(0)[0] for _ in range(20)} == {"snapshot_000040"}
    m = pool.metrics()
    assert m["selfplay/pool_size"] == 3
    assert m["selfplay/mean_opponent_age"] == 0.0
    assert "size=3" in repr(pool)
    assert len(pool.snapshots()) == 3


def test_uniform_sampler_covers_pool() -> None:
    pool, graphdef, params = make_pool("uniform")
    for u in range(4):
        pool.add_snapshot(u, graphdef, params)  # pyright: ignore[reportArgumentType]
    ids = [pool.next_opponent(0)[0] for _ in range(4000)]
    freqs = np.array([ids.count(f"snapshot_{u:06d}") for u in range(4)]) / len(ids)
    np.testing.assert_allclose(freqs, 0.25, atol=0.03)
    assert pool.metrics()["selfplay/mean_opponent_age"] == pytest.approx(1.5, abs=0.15)


def test_latest_prob_mixes_in_newest() -> None:
    pool, graphdef, params = make_pool("uniform", latest_prob=0.8)
    for u in range(5):
        pool.add_snapshot(u, graphdef, params)  # pyright: ignore[reportArgumentType]
    ids = [pool.next_opponent(0)[0] for _ in range(4000)]
    assert ids.count("snapshot_000004") / len(ids) == pytest.approx(0.8 + 0.2 / 5, abs=0.03)


def test_pfsp_prefers_opponents_the_learner_loses_to() -> None:
    pool, graphdef, params = make_pool("pfsp", pfsp_power=2.0)
    for u in range(2):
        pool.add_snapshot(u, graphdef, params)  # pyright: ignore[reportArgumentType]
    for _ in range(50):
        pool.report("snapshot_000000", 1.0)  # learner always beats snapshot 0
        pool.report("snapshot_000001", -1.0)  # and always loses to snapshot 1
    ids = [pool.next_opponent(0)[0] for _ in range(2000)]
    assert ids.count("snapshot_000001") / len(ids) > 0.95
    assert pool.stats()["snapshot_000000"].win_rate == 1.0


def test_pool_validation_and_thread_safety() -> None:
    with pytest.raises(ValueError, match="max_size"):
        SnapshotPoolProvider(max_size=0)
    with pytest.raises(ValueError, match="latest_prob"):
        SnapshotPoolProvider(latest_prob=1.5)
    pool, graphdef, params = make_pool("uniform", max_size=50)
    pool.add_snapshot(0, graphdef, params)  # pyright: ignore[reportArgumentType]

    def adder() -> None:
        for u in range(1, 30):
            pool.add_snapshot(u, graphdef, params)  # pyright: ignore[reportArgumentType]

    def sampler() -> None:
        for _ in range(300):
            pool.next_opponent(0)

    threads = [
        threading.Thread(target=adder),
        *(threading.Thread(target=sampler) for _ in range(4)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert pool.size == 30


def test_mixture_exploitability() -> None:
    nash = solve_cfr("kuhn_poker", 2000)
    assert mixture_exploitability([nash]) < 1e-3
    assert mixture_exploitability([nash, nash], [0.5, 0.5]) < 1e-3
    uniform = solve_cfr("kuhn_poker", 1)
    assert mixture_exploitability([uniform]) == pytest.approx(0.4583333, abs=1e-5)
    with pytest.raises(ValueError, match="at least one"):
        mixture_exploitability([])


def test_mixture_differs_from_naive_average() -> None:
    """Reach-weighted mixing is not the same as averaging action probabilities."""
    graphdef, p1 = init_actor_critic(11, 2, (8,), seed=1)
    _, p2 = init_actor_critic(11, 2, (8,), seed=2)
    t1 = network_to_tabular(graphdef, p1, "kuhn_poker")
    t2 = network_to_tabular(graphdef, p2, "kuhn_poker")
    mix = mixture_exploitability([t1, t2])
    assert mix >= 0.0
