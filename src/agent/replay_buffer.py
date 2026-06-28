"""
Experience Replay Buffer for the DRRN agent.

Because the action space is dynamic (different valid actions at every step),
each transition stores the *full candidate list of the next state* alongside
the standard (S, A, R, S', done) tuple. This is required to compute the
max-Q target in the TD update:

    y_i = R_i + γ · max_{a' ∈ Candidates_{i+1}} Q(S'_i, a'; θ_target)

Without next_candidates we cannot compute the target, so they are a
first-class field in every stored transition.
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass
class Transition:
    state: str                         # augmented state text S_t
    action: str                        # chosen action A_t
    reward: float                      # immediate reward R_t
    next_state: str                    # augmented state text S_{t+1}
    done: bool                         # terminal flag
    next_candidates: List[str] = field(default_factory=list)
    # next_candidates is empty when done=True (terminal states have no successors)


class ReplayBuffer:
    """
    Uniform random experience replay buffer backed by a deque.

    The deque automatically evicts the oldest transitions once capacity
    is reached, keeping memory bounded.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self._buffer: deque = deque(maxlen=capacity)
        self.capacity = capacity

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def push(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        done: bool,
        next_candidates: Optional[List[str]] = None,
    ) -> None:
        self._buffer.append(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                next_candidates=next_candidates or [],
            )
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> List[Transition]:
        """Return a random mini-batch of transitions."""
        return random.sample(self._buffer, batch_size)

    def can_sample(self, batch_size: int) -> bool:
        return len(self._buffer) >= batch_size

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self)}/{self.capacity})"
