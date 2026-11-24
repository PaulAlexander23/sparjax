import numpy as np
import pyspiel
import pytest
from pettingzoo.test import api_test

from sparjax.envs.openspiel_aec import OpenSpielAECEnv

GAMES = ["kuhn_poker", "leduc_poker"]


def play_random(env: OpenSpielAECEnv, seed: int) -> tuple[list[int], dict[str, float]]:
    """Play one game with uniformly random legal actions; return actions and total rewards."""
    rng = np.random.default_rng(seed + 1000)
    env.reset(seed=seed)
    actions: list[int] = []
    totals = dict.fromkeys(env.possible_agents, 0.0)
    for agent in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        assert obs is not None
        totals[agent] += reward
        if term or trunc:
            env.step(None)
            continue
        legal = np.flatnonzero(obs["action_mask"])
        action = int(rng.choice(legal))
        actions.append(action)
        env.step(action)
    return actions, totals


# PettingZoo's api_test warns about any dict observation (space and value), including
# the {"observation", "action_mask"} convention its own classic envs use.
@pytest.mark.filterwarnings("ignore:Observation space for each agent probably should be")
@pytest.mark.filterwarnings("ignore:Observation is not a NumPy array")
@pytest.mark.parametrize("game", GAMES)
def test_pettingzoo_api(game: str) -> None:
    api_test(OpenSpielAECEnv(game), num_cycles=200)


@pytest.mark.parametrize("game", GAMES)
def test_same_seed_same_game(game: str) -> None:
    env = OpenSpielAECEnv(game)
    assert play_random(env, seed=7) == play_random(env, seed=7)


@pytest.mark.parametrize("game", GAMES)
def test_different_seeds_deal_differently(game: str) -> None:
    env = OpenSpielAECEnv(game)
    deals = set()
    for seed in range(20):
        env.reset(seed=seed)
        deals.add(str(env.game_state))
    assert len(deals) > 1


@pytest.mark.parametrize("game", GAMES)
def test_returns_match_pyspiel_and_are_zero_sum(game: str) -> None:
    env = OpenSpielAECEnv(game)
    for seed in range(200):
        _, totals = play_random(env, seed)
        expected = env.game_state.returns()
        assert totals["player_0"] == pytest.approx(expected[0])
        assert totals["player_1"] == pytest.approx(expected[1])
        assert sum(totals.values()) == pytest.approx(0.0)


@pytest.mark.parametrize("game", GAMES)
def test_mask_and_info_state_follow_turn(game: str) -> None:
    env = OpenSpielAECEnv(game)
    env.reset(seed=0)
    state = env.game_state
    mover = env.agent_selection
    other = next(a for a in env.possible_agents if a != mover)

    mover_obs = env.observe(mover)
    assert np.flatnonzero(mover_obs["action_mask"]).tolist() == state.legal_actions()
    assert not env.observe(other)["action_mask"].any()

    player = env.player_index(mover)
    assert env.infos[mover]["info_state"] == state.information_state_string(player)
    np.testing.assert_allclose(
        mover_obs["observation"], np.asarray(state.information_state_tensor(player))
    )


def test_illegal_action_raises() -> None:
    env = OpenSpielAECEnv("leduc_poker")
    env.reset(seed=0)
    legal = env.game_state.legal_actions()
    illegal = next(a for a in range(env.game.num_distinct_actions()) if a not in legal)
    with pytest.raises(ValueError, match="illegal action"):
        env.step(illegal)


def test_none_action_for_live_agent_raises() -> None:
    env = OpenSpielAECEnv("kuhn_poker")
    env.reset(seed=0)
    with pytest.raises(ValueError, match="must not be None"):
        env.step(None)


def test_render_ansi() -> None:
    env = OpenSpielAECEnv("kuhn_poker", render_mode="ansi")
    env.reset(seed=0)
    assert env.render() == str(env.game_state)
    assert OpenSpielAECEnv("kuhn_poker").render() is None


def test_metadata_name_mentions_game() -> None:
    assert OpenSpielAECEnv("leduc_poker").metadata["name"] == "openspiel_leduc_poker_v0"


@pytest.mark.parametrize(
    ("game", "kwargs", "match"),
    [
        ("kuhn_poker", {"players": 3}, "expected 2 players"),
        ("matrix_rps", {}, "sequential"),
        ("kuhn_poker", {"render_mode": "human"}, "render_mode"),
    ],
)
def test_rejects_unsupported(game: str, kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        OpenSpielAECEnv(game, **kwargs)  # pyright: ignore[reportArgumentType]


def test_rejects_non_zero_sum() -> None:
    # Pick any registered two-player sequential general-sum game with an info-state tensor.
    for game_type in pyspiel.registered_games():
        if (
            game_type.utility == pyspiel.GameType.Utility.GENERAL_SUM
            and game_type.dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
            and game_type.provides_information_state_tensor
            and game_type.min_num_players <= 2 <= game_type.max_num_players
            and game_type.default_loadable
        ):
            try:
                pyspiel.load_game(game_type.short_name, {"players": 2})
            except pyspiel.SpielError:
                continue
            with pytest.raises(ValueError, match="zero-sum"):
                OpenSpielAECEnv(game_type.short_name, players=2)
            return
    pytest.skip("no suitable general-sum game registered")


def test_rejects_game_without_info_state_tensor() -> None:
    for game_type in pyspiel.registered_games():
        if (
            game_type.utility == pyspiel.GameType.Utility.ZERO_SUM
            and game_type.dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
            and not game_type.provides_information_state_tensor
            and game_type.min_num_players <= 2 <= game_type.max_num_players
            and game_type.default_loadable
        ):
            with pytest.raises(ValueError, match="information_state_tensor"):
                OpenSpielAECEnv(game_type.short_name)
            return
    pytest.skip("no suitable game registered")
