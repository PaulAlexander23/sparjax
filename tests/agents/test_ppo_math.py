import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from sparjax.agents.ppo.loss import Batch, PPOHyperparams, compute_gae, make_update_fn, ppo_loss
from sparjax.agents.ppo.networks import MaskedCategorical, apply, init_actor_critic


def numpy_gae(
    r: np.ndarray, v: np.ndarray, d: np.ndarray, last_v: np.ndarray, gamma: float, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    T = r.shape[0]
    adv = np.zeros_like(r)
    nxt = np.zeros_like(last_v)
    for t in reversed(range(T)):
        next_v = last_v if t == T - 1 else v[t + 1]
        nd = 1.0 - d[t]
        delta = r[t] + gamma * next_v * nd - v[t]
        nxt = delta + gamma * lam * nd * nxt
        adv[t] = nxt
    return adv, adv + v


@pytest.mark.parametrize(("gamma", "lam"), [(1.0, 0.95), (0.9, 1.0), (0.99, 0.0)])
def test_gae_matches_reference(gamma: float, lam: float) -> None:
    rng = np.random.default_rng(0)
    T, N = 7, 5
    r = rng.normal(size=(T, N)).astype(np.float32)
    v = rng.normal(size=(T, N)).astype(np.float32)
    d = rng.random((T, N)) < 0.3
    last_v = rng.normal(size=N).astype(np.float32)
    adv, ret = compute_gae(r, v, d, last_v, gamma, lam)
    ref_adv, ref_ret = numpy_gae(r, v, d.astype(np.float32), last_v, gamma, lam)
    np.testing.assert_allclose(adv, ref_adv, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret, ref_ret, rtol=1e-5, atol=1e-5)


def test_masked_categorical_respects_mask() -> None:
    logits = jnp.array([[5.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    mask = jnp.array([[False, True, True], [True, True, False]])
    dist = MaskedCategorical(jnp.where(mask, logits, -1e9), mask)
    samples = jax.vmap(dist.sample)(jax.random.split(jax.random.key(0), 2000))
    assert not np.any(np.asarray(samples)[:, 0] == 0)
    assert not np.any(np.asarray(samples)[:, 1] == 2)
    np.testing.assert_allclose(dist.entropy()[1], np.log(2), rtol=1e-6)
    np.testing.assert_allclose(
        jnp.exp(dist.log_prob(jnp.array([1, 0]))), [np.e / (np.e + 1), 0.5], rtol=1e-5
    )
    assert np.isfinite(np.asarray(dist.entropy())).all()


def test_network_masks_logits_and_shapes() -> None:
    graphdef, params = init_actor_critic(4, 3, (8,), seed=0)
    obs = jnp.ones((2, 4))
    mask = jnp.array([[True, False, True], [True, True, True]])
    logits, values = apply(graphdef, params, obs, mask)
    assert logits.shape == (2, 3)
    assert values.shape == (2,)
    assert float(logits[0, 1]) == pytest.approx(-1e9)


def make_batch(T: int = 8, N: int = 4, obs_dim: int = 4, A: int = 3, seed: int = 0) -> Batch:
    rng = np.random.default_rng(seed)
    mask = np.ones((T, N, A), bool)
    mask[..., 2] = rng.random((T, N)) < 0.5
    return Batch(
        obs=rng.normal(size=(T, N, obs_dim)).astype(np.float32),
        mask=mask,
        actions=rng.integers(0, 2, size=(T, N)).astype(np.int32),
        log_probs=np.full((T, N), np.log(1 / 3), np.float32),
        values=np.zeros((T, N), np.float32),
        rewards=rng.normal(size=(T, N)).astype(np.float32),
        dones=rng.random((T, N)) < 0.2,
        last_value=np.zeros(N, np.float32),
    )


def test_ppo_loss_on_policy_has_no_clipping() -> None:
    graphdef, params = init_actor_critic(4, 3, (8,), seed=0)
    b = make_batch()
    obs, mask, actions = (
        jnp.asarray(x).reshape(-1, *x.shape[2:]) for x in (b.obs, b.mask, b.actions)
    )
    logits, _ = apply(graphdef, params, obs, mask)
    old_lp = MaskedCategorical(logits, mask).log_prob(actions)
    adv = jnp.asarray(np.random.default_rng(1).normal(size=old_lp.shape), jnp.float32)
    _, m = ppo_loss(graphdef, params, obs, mask, actions, old_lp, adv, adv, PPOHyperparams())
    assert float(m["ppo/clip_frac"]) == 0.0
    assert abs(float(m["ppo/approx_kl"])) < 1e-6


def test_update_reduces_value_loss() -> None:
    graphdef, params = init_actor_critic(4, 3, (16,), seed=0)
    hp = PPOHyperparams(epochs=8, num_minibatches=2, ent_coef=0.0)
    optimizer = optax.adam(3e-3)
    update = jax.jit(make_update_fn(graphdef, optimizer, hp))
    opt_state = optimizer.init(params)
    batch = make_batch()
    losses: list[float] = []
    metrics: dict[str, jax.Array] = {}
    for i in range(10):
        params, opt_state, metrics = update(params, opt_state, batch, jax.random.key(i))
        losses.append(float(metrics["loss/value"]))
    assert losses[-1] < losses[0]
    assert set(metrics) >= {
        "loss/policy",
        "loss/entropy",
        "ppo/approx_kl",
        "value/explained_variance",
    }


def test_update_rejects_indivisible_minibatches() -> None:
    graphdef, params = init_actor_critic(4, 3, (8,), seed=0)
    optimizer = optax.sgd(0.1)
    update = make_update_fn(graphdef, optimizer, PPOHyperparams(num_minibatches=3))
    with pytest.raises(ValueError, match="divisible"):
        update(params, optimizer.init(params), make_batch(T=2, N=4), jax.random.key(0))


def test_cnn_torso_shapes_and_validation() -> None:
    graphdef, params = init_actor_critic((5, 5, 3), 26, (16,), seed=0, conv_channels=(4, 8))
    obs = jnp.zeros((2, 3, 5, 5, 3), bool)  # extra batch dims, boolean planes
    logits, values = apply(graphdef, params, obs, jnp.ones((2, 3, 26), bool))
    assert logits.shape == (2, 3, 26)
    assert values.shape == (2, 3)
    with pytest.raises(ValueError, match="H x W x C"):
        init_actor_critic(10, 3, (8,), seed=0, conv_channels=(4,))


def test_mlp_flattens_multidimensional_observations() -> None:
    graphdef, params = init_actor_critic((2, 3), 4, (8,), seed=0)
    logits, values = apply(graphdef, params, jnp.ones((7, 2, 3)), jnp.ones((7, 4), bool))
    assert logits.shape == (7, 4)
    assert values.shape == (7,)
