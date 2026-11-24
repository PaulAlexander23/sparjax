import threading
import time

import jax
import jax.numpy as jnp
import pytest

from sparjax.arch import (
    Lifetime,
    OnPolicyPipeline,
    ParameterServer,
    StopRequestedError,
    resolve_device,
)


def test_resolve_device() -> None:
    assert resolve_device("cpu").platform == "cpu"
    assert resolve_device("default") in jax.devices()
    with pytest.raises(ValueError, match="no cpu device"):
        resolve_device("cpu", index=99)


def test_lifetime_captures_first_error_and_stops() -> None:
    life = Lifetime()

    def boom() -> None:
        raise ValueError("bad actor")

    def waiter() -> None:
        while not life.wait(0.01):
            pass

    life.spawn(waiter, "waiter")
    life.spawn(boom, "boom")
    assert not life.join(timeout=5)
    assert life.stopped
    assert isinstance(life.error, ValueError)
    with pytest.raises(RuntimeError, match="worker thread failed"):
        life.raise_if_failed()


def test_lifetime_stop_requested_is_not_an_error() -> None:
    life = Lifetime()

    def stopper() -> None:
        life.stop()
        life.check()

    life.spawn(stopper, "s")
    life.join()
    assert life.error is None
    life.raise_if_failed()
    with pytest.raises(StopRequestedError):
        life.check()


def test_parameter_server_versions_and_blocking() -> None:
    life = Lifetime()
    cpu = resolve_device("cpu")
    server = ParameterServer([cpu, cpu], life)
    assert server.version == -1
    server.publish({"w": jnp.ones(2)}, version=0)
    params, version = server.get(cpu, min_version=0)
    assert version == 0
    assert float(params["w"].sum()) == 2.0
    with pytest.raises(ValueError, match="not newer"):
        server.publish({"w": jnp.zeros(2)}, version=0)

    got: list[int] = []

    def reader() -> None:
        got.append(server.get(cpu, min_version=2)[1])

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)
    assert not got  # still waiting for version 2
    server.publish({"w": jnp.zeros(2)}, version=2)
    t.join(2)
    assert got == [2]


def test_parameter_server_wait_aborts_on_stop() -> None:
    life = Lifetime()
    server = ParameterServer([resolve_device("cpu")], life)
    life.stop()
    with pytest.raises(StopRequestedError):
        server.get(resolve_device("cpu"), min_version=0, poll=0.01)


def test_pipeline_collects_one_per_actor_in_order() -> None:
    life = Lifetime()
    pipe = OnPolicyPipeline(3, life)
    assert pipe.num_actors == 3

    def actor(i: int) -> None:
        for r in range(2):
            pipe.send(i, (i, r), poll=0.01)

    for i in range(3):
        life.spawn(actor, f"a{i}", i)
    assert pipe.collect(poll=0.01) == [(0, 0), (1, 0), (2, 0)]
    assert pipe.collect(poll=0.01) == [(0, 1), (1, 1), (2, 1)]
    assert not life.join()


def test_pipeline_blocking_calls_abort_on_stop() -> None:
    life = Lifetime()
    pipe = OnPolicyPipeline(1, life, maxsize=1)
    pipe.send(0, "first")
    stopper = threading.Timer(0.05, life.stop)
    stopper.start()
    with pytest.raises(StopRequestedError):
        pipe.send(0, "second", poll=0.01)  # queue full until stop
    with pytest.raises(StopRequestedError):
        OnPolicyPipeline(1, life).collect(poll=0.01)
    with pytest.raises(ValueError, match="num_actors"):
        OnPolicyPipeline(0, life)
