"""
Module 4: The Execution Policy — LLM-Biased ε-Greedy

Standard ε-greedy randomly selects any action during exploration. Here,
the random draw is *not uniform*. Instead it samples from the categorical
distribution P provided by the LLM, so exploration is steered toward
linguistically plausible actions even before the Q-network has learned.

Exploit  (x > ε)  : argmax Q(S, a_k)          — pure Q-network
Explore  (x ≤ ε)  : sample from P              — LLM-guided random
"""

import logging
from typing import List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BiasedEpsilonGreedyPolicy:
    """
    Parameters
    ----------
    epsilon_start : float   Initial exploration rate (usually 1.0).
    epsilon_end   : float   Minimum exploration rate.
    epsilon_decay : float   Per-episode multiplicative decay factor.
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        q_values: torch.Tensor,
        candidates: List[str],
        llm_probs: np.ndarray,
        training: bool = True,
    ) -> Tuple[str, int]:
        """
        Returns the chosen (action_string, index_into_candidates).

        Parameters
        ----------
        q_values    : (K,) tensor of Q-values from the DRRN
        candidates  : list of K action strings
        llm_probs   : (K,) numpy array — softmax distribution from the LLM
        training    : if False, always exploit (greedy)
        """
        assert len(candidates) > 0, "Candidate list must not be empty."
        K = len(candidates)

        # Ensure q_values and llm_probs have matching length
        q_np = q_values.cpu().numpy()[:K]
        probs = llm_probs[:K]
        probs = probs / (probs.sum() + 1e-8)  # re-normalise to be safe

        if training and np.random.random() < self.epsilon:
            # --- LLM-biased exploration ---
            idx = int(np.random.choice(K, p=probs))
        else:
            # --- Q-value exploitation ---
            idx = int(np.argmax(q_np))

        return candidates[idx], idx

    # ------------------------------------------------------------------
    # Epsilon scheduling
    # ------------------------------------------------------------------

    def decay(self) -> None:
        """Call once per episode after the episode ends."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    @property
    def current_epsilon(self) -> float:
        return self.epsilon

    def __repr__(self) -> str:
        return (
            f"BiasedEpsilonGreedyPolicy("
            f"ε={self.epsilon:.4f}, "
            f"ε_min={self.epsilon_end}, "
            f"decay={self.epsilon_decay})"
        )
