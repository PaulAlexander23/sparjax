from pathlib import Path

import numpy as np
import pytest

from sparjax.envs.openspiel_aec import OpenSpielAECEnv
from sparjax.opponents import OpponentBatch, TabularPolicy, TabularPolicyOpponent
from sparjax.opponents.cfr import best_response_report, solve_cfr, to_openspiel_policy

KUHN_VALUE_SEAT0 = -1 / 18
# Exploitability of the uniform random policy in Kuhn poker (OpenSpiel's own tests).
KUHN_UNIFORM_EXPLOITABILITY = 0.458333333


@pytest.fixture(scope="module")
def kuhn_cfr() -> TabularPolicy:
    return solve_cfr("kuhn_poker", 2000)


def test_kuhn_cfr_reaches_game_value(kuhn_cfr: TabularPolicy) -> None:
    rep = best_response_report(kuhn_cfr)
    assert len(kuhn_cfr) == 12
    assert rep.on_policy_value[0] == pytest.approx(KUHN_VALUE_SEAT0, abs=1e-3)
    assert rep.on_policy_value[1] == pytest.approx(-KUHN_VALUE_SEAT0, abs=1e-3)
    assert rep.exploitability < 1e-3
    assert rep.nash_conv == pytest.approx(2 * rep.exploitability)
    assert rep.mean_best_response == pytest.approx(rep.exploitability, abs=1e-9)


def test_uniform_policy_exploitability() -> None:
    empty = TabularPolicy("kuhn_poker", 2, {})  # missing states -> uniform
    rep = best_response_report(empty)
    assert rep.exploitability == pytest.approx(KUHN_UNIFORM_EXPLOITABILITY, abs=1e-6)


@pytest.mark.parametrize("algorithm", ["cfr", "cfr_plus"])
def test_more_iterations_less_exploitable(algorithm: str) -> None:
    few = best_response_report(solve_cfr("kuhn_poker", 5, algorithm=algorithm))  # pyright: ignore[reportArgumentType]
    many = best_response_report(solve_cfr("kuhn_poker", 500, algorithm=algorithm))  # pyright: ignore[reportArgumentType]
    assert many.exploitability < few.exploitability


def test_metadata_and_round_trip(kuhn_cfr: TabularPolicy, tmp_path: Path) -> None:
    assert kuhn_cfr.metadata["solver"] == "open_spiel.cfr_plus"
    assert kuhn_cfr.metadata["iterations"] == 2000
    loaded = TabularPolicy.load(kuhn_cfr.save(tmp_path / "k.npz"))
    assert best_response_report(loaded) == best_response_report(kuhn_cfr)


def test_to_openspiel_policy_copies_rows(kuhn_cfr: TabularPolicy) -> None:
    os_pol = to_openspiel_policy(kuhn_cfr)
    for key, idx in os_pol.state_lookup.items():
        np.testing.assert_allclose(os_pol.action_probability_array[idx], kuhn_cfr.probs[key])


def test_invalid_iterations() -> None:
    with pytest.raises(ValueError, match="iterations"):
        solve_cfr("kuhn_poker", 0)


def test_non_two_player_rejected() -> None:
    three = TabularPolicy("kuhn_poker", 2, {}, metadata={"game_params": {"players": 3}})
    with pytest.raises(ValueError, match="two-player"):
        best_response_report(three)


@pytest.mark.parametrize("game", ["kuhn_poker", "leduc_poker"])
def test_env_info_states_are_policy_keys(game: str) -> None:
    """The env's info_state strings must index the CFR table for every reachable state."""
    policy = solve_cfr(game, 1)
    opponent = TabularPolicyOpponent(policy)  # on_missing="error"
    env = OpenSpielAECEnv(game)
    rng = np.random.default_rng(0)
    for seed in range(300):
        env.reset(seed=seed)
        for _agent in env.agent_iter():
            obs, _, term, trunc, info = env.last()
            assert obs is not None
            if term or trunc:
                env.step(None)
                continue
            batch = OpponentBatch(
                obs["observation"][None], obs["action_mask"][None], [info["info_state"]]
            )
            env.step(int(opponent.act(batch, rng)[0]))
