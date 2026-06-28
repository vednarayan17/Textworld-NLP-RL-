"""
Module 3: The Evaluator — Deep Reinforcement Relevance Network (DRRN)

Architecture
------------
1. Frozen sentence transformer  → raw dense embedding (no grad)
2. Shared trainable projection  → maps to RL latent space (grads flow here)
3. Dot-product interaction       → Q(S, A) = h_s · h_a

Because the projection is *shared* between state and action encoding, both
are mapped into the same semantic space before computing similarity.

Only the projection layer's parameters (θ) are updated during training.
The sentence transformer stays frozen throughout.

Q-value semantics: higher Q ≈ higher expected cumulative reward from state S
when action A is taken.
"""

import copy
import logging
from typing import List

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class DRRN(nn.Module):
    """
    Deep Reinforcement Relevance Network.

    Parameters
    ----------
    sentence_model_name : str
        Name of the SentenceTransformer model (e.g. 'all-MiniLM-L6-v2').
    projection_dim : int
        Output dimension of the shared trainable projection layer.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        sentence_model_name: str = "all-MiniLM-L6-v2",
        projection_dim: int = 128,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self._device = torch.device(device)

        # ---- Frozen sentence transformer --------------------------------
        logger.info("Loading sentence transformer: %s", sentence_model_name)
        self.sentence_encoder = SentenceTransformer(sentence_model_name)
        self.sentence_encoder = self.sentence_encoder.to(self._device)
        for param in self.sentence_encoder.parameters():
            param.requires_grad = False
        self.sentence_encoder.eval()

        embedding_dim: int = self.sentence_encoder.get_sentence_embedding_dimension()
        logger.info("Sentence embedding dim: %d  →  projection dim: %d", embedding_dim, projection_dim)

        # ---- Shared trainable projection ---------------------------------
        # Used identically for both state and action encoding.
        # A single Linear layer keeps the parameter count small and makes the
        # dot-product interaction well-conditioned.
        self.projection = nn.Linear(embedding_dim, projection_dim)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        self.to(self._device)

    # ------------------------------------------------------------------
    # Core encode
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str]) -> torch.Tensor:
        """
        Returns projected embeddings of shape (len(texts), projection_dim).

        The sentence encoder forward pass runs inside torch.no_grad() so its
        output is a leaf tensor. Gradients then flow through self.projection
        during the backward pass, updating only those weights.
        """
        with torch.no_grad():
            raw: torch.Tensor = self.sentence_encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=64,
            )
        raw = raw.to(self._device)
        return self.projection(raw)  # (N, projection_dim) — has grad w.r.t. projection

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def compute_q_batch(
        self,
        states: List[str],
        actions: List[str],
    ) -> torch.Tensor:
        """
        Compute Q(S_i, A_i) for a batch of (state, action) pairs.

        Gradients flow through the projection layer, enabling TD updates.

        Returns: shape (B,)
        """
        assert len(states) == len(actions), "states and actions must be the same length"
        B = len(states)

        # Single encoder call for all 2B texts — efficient
        h_all = self._encode(states + actions)  # (2B, d)
        h_s = h_all[:B]                          # (B, d)
        h_a = h_all[B:]                          # (B, d)
        return (h_s * h_a).sum(dim=-1)           # (B,)

    # ------------------------------------------------------------------
    # Inference / target computation (no gradient)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def q_values_for_candidates(
        self,
        state_text: str,
        candidates: List[str],
    ) -> torch.Tensor:
        """
        Compute Q(S, a_k) for all K candidate actions.

        Used both during action selection and for target max-Q computation.

        Returns: shape (K,)
        """
        if not candidates:
            return torch.tensor([], device=self._device)

        h_all = self._encode([state_text] + candidates)  # (1+K, d)
        h_s = h_all[:1]                                   # (1, d)
        h_a = h_all[1:]                                   # (K, d)
        return (h_s * h_a).sum(dim=-1)                   # (K,)

    @torch.no_grad()
    def max_q_for_next_states(
        self,
        next_states: List[str],
        next_candidates_list: List[List[str]],
    ) -> torch.Tensor:
        """
        Batch computation of max_a' Q(S'_i, a') for TD target.

        Flattens all (next_state, candidate) pairs into one encoder call,
        then reshapes and takes max per row.

        Returns: shape (B,)
        """
        B = len(next_states)

        # Build flat lists — state S'_i repeated len(candidates_i) times
        flat_states: List[str] = []
        flat_actions: List[str] = []
        group_sizes: List[int] = []

        for s, cands in zip(next_states, next_candidates_list):
            k = len(cands)
            flat_states.extend([s] * k)
            flat_actions.extend(cands)
            group_sizes.append(k)

        if not flat_states:
            return torch.zeros(B, device=self._device)

        # One big encode call for all flat_states + flat_actions
        total = len(flat_states)
        h_all = self._encode(flat_states + flat_actions)  # (2*total, d)
        h_s_flat = h_all[:total]                           # (total, d)
        h_a_flat = h_all[total:]                           # (total, d)
        q_flat = (h_s_flat * h_a_flat).sum(dim=-1)        # (total,)

        # Split and take max within each group
        max_q = torch.zeros(B, device=self._device)
        offset = 0
        for i, k in enumerate(group_sizes):
            if k > 0:
                max_q[i] = q_flat[offset : offset + k].max()
            offset += k

        return max_q  # (B,)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Return only the parameters that should be optimised."""
        return self.projection.parameters()

    def get_embedding_dim(self) -> int:
        return self.sentence_encoder.get_sentence_embedding_dimension()

    def get_projection_dim(self) -> int:
        return self.projection.out_features


def build_target_network(online_drrn: DRRN) -> DRRN:
    """
    Create a frozen copy of the online DRRN for computing TD targets.

    The target network weights are periodically synced with the online
    network (hard update every target_update_freq steps).
    """
    target = copy.deepcopy(online_drrn)
    target.eval()
    for param in target.parameters():
        param.requires_grad = False
    logger.info("Target network created (hard-copy of online DRRN).")
    return target


def sync_target_network(online: DRRN, target: DRRN) -> None:
    """Hard-copy weights from online → target network."""
    target.load_state_dict(online.state_dict())
