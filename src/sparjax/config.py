"""Typed experiment configuration (frozen dataclasses, exposed on the CLI via tyro)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sparjax.agents.ppo.loss import PPOHyperparams
from sparjax.arch.devices import DeviceKind

__all__ = ["ArchConfig", "EnvConfig", "OpponentConfig", "PPOConfig", "TrainConfig"]


@dataclass(frozen=True)
class EnvConfig:
    """Which game to play and how many copies each actor runs."""

    game: str = "kuhn_poker"
    """OpenSpiel game short name (two-player, zero-sum, sequential)."""
    num_envs: int = 64
    """Environments per actor thread."""


@dataclass(frozen=True)
class OpponentConfig:
    """The opponent the learner trains against."""

    kind: Literal["random", "cfr", "policy"] = "random"
    """``random``: uniform over legal actions. ``cfr``: solve with CFR at start-up.
    ``policy``: load a tabular policy written by ``sparjax cfr-solve``."""
    cfr_iters: int = 20
    """CFR iterations when ``kind == "cfr"``."""
    cfr_algorithm: Literal["cfr", "cfr_plus"] = "cfr_plus"
    """CFR variant when ``kind == "cfr"``."""
    policy_path: Path | None = None
    """Tabular policy file when ``kind == "policy"``."""


@dataclass(frozen=True)
class PPOConfig:
    """Network, optimiser and PPO update settings."""

    lr: float = 3e-4
    max_grad_norm: float = 0.5
    rollout_len: int = 16
    """Learner decisions per env per rollout."""
    hidden: tuple[int, ...] = (64, 64)
    update: PPOHyperparams = field(default_factory=PPOHyperparams)


@dataclass(frozen=True)
class ArchConfig:
    """Sebulba layout."""

    num_actors: int = 2
    """Actor threads."""
    actor_device: DeviceKind = "cpu"
    """Where actors run policy inference. Small nets are usually fastest on the CPU."""
    learner_device: DeviceKind = "default"
    """Where the PPO update runs (the GPU when a CUDA jaxlib is installed)."""
    max_lag: int = 1
    """Allowed parameter staleness in updates (0 = strictly on-policy, 1 = Sebulba)."""


@dataclass(frozen=True)
class TrainConfig:
    """Top-level training configuration."""

    env: EnvConfig = field(default_factory=EnvConfig)
    opponent: OpponentConfig = field(default_factory=OpponentConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    arch: ArchConfig = field(default_factory=ArchConfig)
    total_timesteps: int = 500_000
    """Learner decisions across all actors and envs."""
    seed: int = 0
    log_every: int = 10
    """Log every this many updates."""
    log_dir: Path | None = None
    """If set, metrics are also written to ``<log_dir>/metrics.jsonl``."""

    @property
    def steps_per_update(self) -> int:
        """Learner decisions consumed by one update."""
        return self.arch.num_actors * self.env.num_envs * self.ppo.rollout_len

    @property
    def num_updates(self) -> int:
        """Number of learner updates implied by ``total_timesteps``."""
        return max(1, self.total_timesteps // self.steps_per_update)
