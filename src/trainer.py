"""Training loop: MSE value loss + cross-entropy policy loss, Adam with cosine decay."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from replay_buffer import ReplayBuffer


class Trainer:
    """Wraps model + optimizer and runs gradient updates from a ReplayBuffer.

    Policy loss  — cross-entropy with soft MCTS targets:  −Σ π·log p
    Value  loss  — mean-squared error:                    (v − z)²
    Total  loss  — unweighted sum of both heads
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size

        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    # ------------------------------------------------------------------
    # Core gradient step
    # ------------------------------------------------------------------

    def _step(
        self,
        states: "np.ndarray",
        policies: "np.ndarray",
        values: "np.ndarray",
    ) -> tuple[float, float, float]:
        """One forward+backward pass. Returns (policy_loss, value_loss, total_loss)."""
        self.model.train()

        s = torch.from_numpy(states).to(self.device)
        p = torch.from_numpy(policies).to(self.device)          # (B, POLICY_SIZE)
        v = torch.from_numpy(values).unsqueeze(1).to(self.device)  # (B, 1)

        self.optimizer.zero_grad()

        pol_logits, val_pred = self.model(s)

        # Cross-entropy with soft targets: −Σ π · log softmax(logits)
        log_probs = F.log_softmax(pol_logits, dim=1)
        policy_loss = -(p * log_probs).sum(dim=1).mean()

        value_loss = F.mse_loss(val_pred, v)

        loss = policy_loss + value_loss
        loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), loss.item()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        buffer: ReplayBuffer,
        num_steps: int,
    ) -> dict[str, float]:
        """Sample from buffer and run num_steps gradient updates.

        A fresh CosineAnnealingLR schedule is applied across the num_steps,
        ending at lr=1e-5.  The optimizer's base lr is preserved for the
        next call.

        Returns:
            Dict with average "loss", "policy_loss", "value_loss" over all steps.
        """
        if not buffer.is_ready(self.batch_size):
            raise ValueError(
                f"Buffer has {len(buffer)} examples; need at least {self.batch_size}."
            )

        # Cosine schedule scoped to this training call
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_steps, eta_min=1e-5
        )

        total_pol = total_val = total_loss = 0.0

        for _ in tqdm(range(num_steps), desc="train", leave=False):
            states, policies, values = buffer.sample(self.batch_size)
            pl, vl, l = self._step(states, policies, values)
            total_pol  += pl
            total_val  += vl
            total_loss += l
            scheduler.step()

        n = num_steps
        return {
            "loss":        total_loss / n,
            "policy_loss": total_pol  / n,
            "value_loss":  total_val  / n,
        }

    # ------------------------------------------------------------------
    # Checkpointing (model + optimiser state for resuming training)
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        """Save model weights + optimiser state to path."""
        torch.save(
            {
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                # Preserve architecture config stored on ChessNet
                "num_blocks":  getattr(self.model, "_num_blocks",  None),
                "num_filters": getattr(self.model, "_num_filters", None),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        """Restore model weights + optimiser state from path."""
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data["model_state"])
        self.optimizer.load_state_dict(data["optimizer_state"])
