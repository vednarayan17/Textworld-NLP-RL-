"""
Module 1: Temporal State Manager (Action-History Buffer)

TextWorld is partially observable. This module acts as the agent's short-term
memory by maintaining a rolling window of (action, observation) pairs so that
the encoded state carries causal context, not just the latest room description.

Buffer format example (window_size=3, after 2 transitions):
  [Previous] open door -> The door is locked.
  [Previous] go north  -> You are in a dark corridor.
  [Current] You see a key on the floor.
"""

from collections import deque
from typing import List, Tuple


class ActionHistoryBuffer:
    def __init__(self, window_size: int = 3):
        self._window_size = window_size
        # Each element: (action_taken, resulting_observation)
        self._buffer: deque = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, initial_obs: str) -> None:
        """Call at the start of every episode with the first observation."""
        self._buffer.clear()
        self._buffer.append(("[Start]", initial_obs))

    def update(self, action: str, observation: str) -> None:
        """Append a new transition after each env.step()."""
        self._buffer.append((action, observation))

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def get_state_text(self) -> str:
        """
        Returns the augmented state S_t as a single string.

        All but the last entry are tagged [Previous action -> observation].
        The last entry is tagged [Current observation].
        """
        items: List[Tuple[str, str]] = list(self._buffer)
        parts: List[str] = []

        for i, (action, obs) in enumerate(items):
            if i < len(items) - 1:
                parts.append(f"[Previous] {action} -> {obs.strip()}")
            else:
                # Most recent observation is the "current" state
                parts.append(f"[Current] {obs.strip()}")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ActionHistoryBuffer(window={self._window_size}, filled={len(self._buffer)})"
