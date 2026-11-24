"""Opponents the learner can train against, and providers that choose between them."""

from sparjax.opponents.base import (
    Opponent,
    OpponentBatch,
    OpponentId,
    OpponentProvider,
    sample_actions,
)
from sparjax.opponents.providers import FixedProvider, MixtureEntry, MixtureProvider, OutcomeStats
from sparjax.opponents.simple import RandomOpponent, ScriptedOpponent
from sparjax.opponents.tabular import TabularPolicy, TabularPolicyOpponent

__all__ = [
    "FixedProvider",
    "MixtureEntry",
    "MixtureProvider",
    "Opponent",
    "OpponentBatch",
    "OpponentId",
    "OpponentProvider",
    "OutcomeStats",
    "RandomOpponent",
    "ScriptedOpponent",
    "TabularPolicy",
    "TabularPolicyOpponent",
    "sample_actions",
]
