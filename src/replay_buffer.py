"""Replay buffer: stores recent self-play experience and samples random minibatches."""

import random
import numpy as np
from collections import deque
from pathlib import Path

from self_play import SelfPlayExample


class ReplayBuffer:
    """Fixed-capacity circular buffer of (state, policy, value) training examples.

    Oldest examples are evicted automatically once capacity is reached.
    """

    def __init__(self, maxlen: int = 500_000):
        self.maxlen = maxlen
        self._buffer: deque[SelfPlayExample] = deque(maxlen=maxlen)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add(self, examples: list[SelfPlayExample]) -> None:
        """Append a list of self-play examples (evicts oldest if full)."""
        self._buffer.extend(examples)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw a random minibatch without replacement.

        Returns:
            states:   float32 (B, NUM_PLANES, 8, 8)
            policies: float32 (B, POLICY_SIZE)
            values:   float32 (B,)

        Raises:
            ValueError if the buffer contains fewer than batch_size examples.
        """
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Buffer has {len(self._buffer)} examples, need {batch_size}."
            )
        batch = random.sample(self._buffer, batch_size)
        states, policies, values = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.stack(policies).astype(np.float32),
            np.array(values, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Capacity helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, min_size: int) -> bool:
        """Return True once the buffer holds at least min_size examples."""
        return len(self._buffer) >= min_size

    def __repr__(self) -> str:
        return f"ReplayBuffer({len(self._buffer)}/{self.maxlen})"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the entire buffer to a compressed .npz file."""
        if not self._buffer:
            return
        states, policies, values = zip(*self._buffer)
        np.savez_compressed(
            path,
            states=np.stack(states).astype(np.float32),
            policies=np.stack(policies).astype(np.float32),
            values=np.array(values, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path, maxlen: int = 500_000) -> "ReplayBuffer":
        """Restore a buffer saved with save(). Trims to maxlen if necessary."""
        buf = cls(maxlen=maxlen)
        data = np.load(path)
        states, policies, values = data["states"], data["policies"], data["values"]
        for s, p, v in zip(states, policies, values):
            buf._buffer.append((s, p, float(v)))
        return buf
