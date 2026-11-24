"""Exact evaluation of a policy network in small OpenSpiel games.

For games with few information states (Kuhn, Leduc) we can query the network at *every*
information state, turn it into a tabular policy, and let OpenSpiel compute exact
quantities. There is no sampling noise:

* the learner's expected payoff against a fixed tabular opponent, per seat;
* the network policy's exploitability (how much a best response would beat it).

Requires the ``openspiel`` extra.
"""
# flax.nnx, optax and jax.lax are only partially typed.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportUnknownLambdaType=false, reportMissingTypeStubs=false

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
import pyspiel
from flax import nnx
from open_spiel.python import policy as os_policy
from open_spiel.python.algorithms import exploitability as os_exploitability
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python.algorithms.expected_game_score import policy_value

from sparjax.agents.ppo.networks import ActorCritic, Params, apply
from sparjax.opponents.cfr import to_openspiel_policy
from sparjax.opponents.tabular import TabularPolicy

__all__ = ["ExactEvaluation", "evaluate_exact", "mixture_exploitability", "network_to_tabular"]


def network_to_tabular(
    graphdef: nnx.GraphDef[ActorCritic], params: Params, game_name: str
) -> TabularPolicy:
    """Query the network at every information state of ``game_name``."""
    game = pyspiel.load_game(game_name)
    template = os_policy.TabularPolicy(game)
    num_actions = int(game.num_distinct_actions())
    obs = np.stack(
        [
            np.asarray(s.information_state_tensor(s.current_player()), np.float32)
            for s in template.states
        ]
    )
    mask = np.zeros((len(template.states), num_actions), bool)
    for i, s in enumerate(template.states):
        mask[i, s.legal_actions()] = True
    logits, _ = jax.jit(apply, static_argnums=0)(graphdef, params, obs, mask)
    probs = np.asarray(jax.nn.softmax(logits, axis=-1), np.float64)
    probs = np.where(mask, probs, 0.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return TabularPolicy(
        game_name=game_name,
        num_actions=num_actions,
        probs={str(k): probs[i] for i, k in enumerate(template.state_lookup)},
        metadata={"source": "network"},
    )


@dataclass(frozen=True, slots=True)
class ExactEvaluation:
    """Exact results for one network policy."""

    payoff_vs_opponent: tuple[float, float]
    """Learner's expected payoff per game in seat 0 and seat 1 against the opponent."""
    exploitability: float
    """Exploitability of the learner's own policy."""

    @property
    def mean_payoff(self) -> float:
        """Expected payoff with seats assigned uniformly at random."""
        return float(np.mean(self.payoff_vs_opponent))


def evaluate_exact(learner: TabularPolicy, opponent: TabularPolicy) -> ExactEvaluation:
    """Exact payoff of ``learner`` against ``opponent`` (both seats) and its exploitability."""
    if learner.game_name != opponent.game_name:
        raise ValueError("learner and opponent policies are for different games")
    game = pyspiel.load_game(learner.game_name)
    lp, op = to_openspiel_policy(learner, game), to_openspiel_policy(opponent, game)
    root = game.new_initial_state()
    # policy_value takes one policy per player.
    seat0 = float(policy_value(root, [lp, op])[0])
    seat1 = float(policy_value(root, [op, lp])[1])
    expl = float(os_exploitability.exploitability(game, lp))
    return ExactEvaluation(payoff_vs_opponent=(seat0, seat1), exploitability=expl)


def mixture_exploitability(
    policies: list[TabularPolicy], weights: list[float] | None = None
) -> float:
    """Exploitability of the *mixture* of ``policies``.

    The mixture picks one of the policies at the start of each game, with probability
    ``weights``. This is the correct average strategy in an extensive-form game. Each policy is
    weighted by its probability of reaching each information state (OpenSpiel's
    ``PolicyAggregator``), not by naively averaging action probabilities state by state.
    In fictitious self-play, this average, not the learner's current policy, is what
    converges to a Nash equilibrium.
    """
    if not policies:
        raise ValueError("need at least one policy")
    game = pyspiel.load_game(policies[0].game_name)
    os_pols = [to_openspiel_policy(p, game) for p in policies]
    w = list(weights) if weights is not None else [1.0 / len(os_pols)] * len(os_pols)
    agg = policy_aggregator.PolicyAggregator(game).aggregate(
        list(range(game.num_players())), [os_pols] * game.num_players(), [w] * game.num_players()
    )
    return float(os_exploitability.exploitability(game, agg))
