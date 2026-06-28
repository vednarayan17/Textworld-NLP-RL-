"""
LLM-DRRN Agent

Orchestrates the five modules defined in the architecture:

  1. ActionHistoryBuffer   — builds augmented state S_t from rolling window
  2. LLMCandidateGenerator — produces K candidates + distribution P
  3. DRRN (online)         — scores candidates via Q(S, A) = h_s · h_a
  4. BiasedEpsilonGreedy   — selects action via Q-values and P
  5. ReplayBuffer + TD     — learns from stored (S, A, R, S', Cands') tuples

The agent owns the optimizer and the target network. The trainer calls
agent.act() and agent.observe() per step, then agent.learn() per update.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.agent.drrn import DRRN, build_target_network, sync_target_network
from src.agent.llm_engine import LLMCandidateGenerator
from src.agent.policy import BiasedEpsilonGreedyPolicy
from src.agent.replay_buffer import ReplayBuffer, Transition
from src.agent.state_manager import ActionHistoryBuffer

logger = logging.getLogger(__name__)


class LLMDRRNAgent:
    """
    Parameters
    ----------
    config : dict
        Full configuration dictionary (merged from config.yaml).
    """

    def __init__(self, config: dict) -> None:
        self._cfg = config
        train_cfg = config["training"]
        llm_cfg = config["llm"]
        drrn_cfg = config["drrn"]
        sm_cfg = config["state_manager"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Agent device: %s", self.device)

        # Module 1
        self.state_manager = ActionHistoryBuffer(sm_cfg["window_size"])

        # Module 2 — frozen LLM (Ollama / HuggingFace / mock)
        self.llm = LLMCandidateGenerator(llm_cfg)

        # Module 3 — online DRRN
        self.drrn = DRRN(
            sentence_model_name=drrn_cfg["sentence_model"],
            projection_dim=drrn_cfg["projection_dim"],
            device=self.device,
        )

        # Target network (frozen copy, periodically synced)
        self.target_drrn = build_target_network(self.drrn)

        # Module 4
        self.policy = BiasedEpsilonGreedyPolicy(
            epsilon_start=train_cfg["epsilon_start"],
            epsilon_end=train_cfg["epsilon_end"],
            epsilon_decay=train_cfg["epsilon_decay"],
        )

        # Module 5 — replay buffer
        self.replay = ReplayBuffer(train_cfg["replay_buffer_capacity"])

        # Optimiser — only projection layer parameters
        self.optimizer = torch.optim.AdamW(
            self.drrn.trainable_parameters(),
            lr=train_cfg["learning_rate"],
        )

        self.gamma: float = train_cfg["gamma"]
        self.batch_size: int = train_cfg["batch_size"]
        self.min_replay: int = train_cfg["min_replay_size"]
        self.target_update_freq: int = train_cfg["target_update_freq"]
        self.grad_clip: float = train_cfg["grad_clip_norm"]

        self._step_count: int = 0
        self._episode_count: int = 0

        # State cached between act() and observe()
        self._last_state_text: Optional[str] = None
        self._last_candidates: Optional[List[str]] = None
        self._last_probs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, initial_obs: str) -> None:
        """Call at the start of every episode."""
        self.state_manager.reset(initial_obs)
        self._last_state_text = None
        self._last_candidates = None
        self._last_probs = None

    # ------------------------------------------------------------------
    # Module 1 + 2 + 3 + 4: Act
    # ------------------------------------------------------------------

    def act(
        self,
        admissible_commands: List[str],
        training: bool = True,
    ) -> str:
        """
        Select an action for the current state.

        1. Build S_t from history buffer
        2. Ask LLM for K candidates + distribution P
        3. Score candidates with DRRN Q-values
        4. Apply biased ε-greedy policy

        Returns the chosen action string.
        """
        state_text = self.state_manager.get_state_text()
        candidates, probs = self.llm.generate_candidates(state_text, admissible_commands)

        q_values = self.drrn.q_values_for_candidates(state_text, candidates)
        action, _ = self.policy.select_action(q_values, candidates, probs, training)

        # Cache for observe()
        self._last_state_text = state_text
        self._last_candidates = candidates
        self._last_probs = probs

        return action

    # ------------------------------------------------------------------
    # Module 1 + 5: Observe and store transition
    # ------------------------------------------------------------------

    def observe(
        self,
        action: str,
        reward: float,
        next_obs: str,
        done: bool,
        next_admissible: List[str],
    ) -> None:
        """
        Record the outcome of the last action.

        1. Update history buffer with (action, next_obs)
        2. Build S_{t+1}
        3. Get next candidates (for TD target) — caches one LLM call
        4. Store transition in replay buffer
        """
        assert self._last_state_text is not None, "observe() called before act()"

        # Update history buffer → S_{t+1}
        self.state_manager.update(action, next_obs)
        next_state_text = self.state_manager.get_state_text()

        # Next-step candidates needed for TD target
        if done or not next_admissible:
            next_candidates: List[str] = []
        else:
            next_candidates, _ = self.llm.generate_candidates(next_state_text, next_admissible)

        self.replay.push(
            state=self._last_state_text,
            action=action,
            reward=reward,
            next_state=next_state_text,
            done=done,
            next_candidates=next_candidates,
        )

        # Clear cache
        self._last_state_text = None

    # ------------------------------------------------------------------
    # Module 5: TD Learning step
    # ------------------------------------------------------------------

    def learn(self) -> Optional[float]:
        """
        Sample a mini-batch from replay and update the DRRN projection layer.

        Returns the loss value, or None if the buffer is not yet full enough.
        """
        if not self.replay.can_sample(self.min_replay):
            return None
        if not self.replay.can_sample(self.batch_size):
            return None

        batch: List[Transition] = self.replay.sample(self.batch_size)
        loss = self._td_update(batch)

        self._step_count += 1

        # Hard-update target network periodically
        if self._step_count % self.target_update_freq == 0:
            sync_target_network(self.drrn, self.target_drrn)
            logger.debug("Target network synced at step %d", self._step_count)

        return loss

    def _td_update(self, batch: List[Transition]) -> float:
        """
        Temporal Difference update:

            y_i = R_i + γ · max_{a'} Q(S'_i, a'; θ_target)  [non-terminal]
            y_i = R_i                                          [terminal]

            L(θ) = (1/B) Σ (y_i - Q(S_i, A_i; θ))²
        """
        states  = [t.state  for t in batch]
        actions = [t.action for t in batch]
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        dones   = torch.tensor([t.done   for t in batch], dtype=torch.float32, device=self.device)

        # ---- Predicted Q(S_i, A_i; θ) — with gradients ----
        q_pred = self.drrn.compute_q_batch(states, actions)  # (B,)

        # ---- Target max Q(S'_i, a'; θ_target) — no gradients ----
        non_terminal_mask = ~torch.tensor([t.done for t in batch], dtype=torch.bool)
        non_terminal_batch = [t for t in batch if not t.done and t.next_candidates]

        max_q_next = torch.zeros(len(batch), device=self.device)

        if non_terminal_batch:
            nt_next_states    = [t.next_state      for t in non_terminal_batch]
            nt_next_cands     = [t.next_candidates for t in non_terminal_batch]
            nt_indices        = [i for i, t in enumerate(batch) if not t.done and t.next_candidates]

            max_q_values = self.target_drrn.max_q_for_next_states(
                nt_next_states, nt_next_cands
            )  # (|non_terminal|,)
            max_q_next[nt_indices] = max_q_values

        # TD target
        y = rewards + self.gamma * max_q_next * (1.0 - dones)  # (B,)

        # ---- Loss and backward ----
        loss = F.mse_loss(q_pred, y.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.drrn.trainable_parameters()), self.grad_clip
        )
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------

    def end_episode(self) -> None:
        """Call at the end of every episode to decay ε."""
        self.policy.decay()
        self._episode_count += 1

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "drrn_state": self.drrn.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epsilon": self.policy.epsilon,
                "step_count": self._step_count,
                "episode_count": self._episode_count,
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.drrn.load_state_dict(ckpt["drrn_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.policy.epsilon = ckpt["epsilon"]
        self._step_count = ckpt.get("step_count", 0)
        self._episode_count = ckpt.get("episode_count", 0)
        sync_target_network(self.drrn, self.target_drrn)
        logger.info("Checkpoint loaded ← %s (ep %d)", path, self._episode_count)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        return {
            "epsilon": self.policy.current_epsilon,
            "replay_size": len(self.replay),
            "step_count": self._step_count,
            "episode_count": self._episode_count,
        }
