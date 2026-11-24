import jax
import jax.numpy as jnp

import sparjax


def test_version_is_exposed() -> None:
    assert isinstance(sparjax.__version__, str)
    assert sparjax.__version__


def test_jax_runs_on_default_backend() -> None:
    assert jax.devices()
    x = jnp.arange(4.0)
    assert float(jnp.sum(x * x)) == 14.0
