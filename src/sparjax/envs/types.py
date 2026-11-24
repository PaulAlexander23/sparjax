"""Shared type aliases for environments."""

from typing import TypedDict

import numpy as np
from jaxtyping import Float, Int8

__all__ = ["AgentID", "Observation"]

AgentID = str


class Observation(TypedDict):
    """Per-agent observation dict (PettingZoo classic convention)."""

    observation: Float[np.ndarray, " obs_dim"]
    action_mask: Int8[np.ndarray, " num_actions"]
