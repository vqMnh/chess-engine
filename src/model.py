"""Residual CNN with a policy head and a value head, à la AlphaZero."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from game import NUM_PLANES, POLICY_SIZE


class _ResBlock(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    """AlphaZero-style residual network.

    Args:
        num_blocks:  Residual blocks in the tower (depth).
        num_filters: Convolutional channels throughout (width).
    """

    def __init__(self, num_blocks: int = 10, num_filters: int = 128):
        super().__init__()
        self._num_blocks  = num_blocks
        self._num_filters = num_filters

        # Input stem: map board planes → filter space
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        # Residual tower
        self.tower = nn.Sequential(*[_ResBlock(num_filters) for _ in range(num_blocks)])

        # Policy head: 1×1 squeeze → flatten → FC to full action space
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_filters, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(2 * 8 * 8, POLICY_SIZE)

        # Value head: 1×1 squeeze → flatten → FC → tanh scalar
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Float tensor (B, NUM_PLANES, 8, 8).
        Returns:
            policy_logits: (B, POLICY_SIZE) — raw logits before masking/softmax.
            value:         (B, 1)           — scalar in (−1, 1).
        """
        x = self.tower(self.stem(x))

        p = self.policy_fc(self.policy_conv(x).flatten(1))
        v = self.value_fc(self.value_conv(x).flatten(1))
        return p, v

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        torch.save({
            "num_blocks":  self._num_blocks,
            "num_filters": self._num_filters,
            "state_dict":  self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> "ChessNet":
        data = torch.load(path, map_location=map_location)
        net  = cls(num_blocks=data["num_blocks"], num_filters=data["num_filters"])
        net.load_state_dict(data["state_dict"])
        return net
