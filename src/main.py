"""Entry point: orchestrates the self-play → train → evaluate loop."""

import copy
import torch
from pathlib import Path

from model import ChessNet
from self_play import generate_games
from replay_buffer import ReplayBuffer
from trainer import Trainer
from evaluator import Evaluator


def train(
    # --- Paths ---
    checkpoints_dir: Path = Path("checkpoints"),
    data_dir:        Path = Path("data"),
    book_path:       Path = Path("books/gm2001.bin"),
    elo_log_path:    Path = Path("runs/elo_log.csv"),
    # --- Architecture ---
    num_blocks:  int = 10,
    num_filters: int = 128,
    # --- Self-play ---
    games_per_iter:  int = 25,
    mcts_sims:       int = 200,
    max_game_moves:  int = 512,
    # --- Training ---
    train_steps:   int = 500,
    batch_size:    int = 256,
    lr:            float = 1e-3,
    weight_decay:  float = 1e-4,
    buffer_maxlen: int = 500_000,
    min_buffer:    int = 1_000,   # wait until this many examples before training
    # --- Evaluation ---
    eval_every:     int = 5,      # evaluate every N iterations
    eval_games:     int = 40,
    eval_sims:      int = 200,
    win_threshold:  float = 0.55,
    # --- Loop ---
    num_iters:  int = 1_000,
    save_every: int = 10,
) -> None:
    """Run the AlphaZero self-play → train → evaluate loop.

    Checkpoints are written to checkpoints_dir; training can be resumed by
    calling train() again with the same directories.

    Loop structure per iteration
    ────────────────────────────
    1. Self-play   — current net plays itself → (state, π, z) examples
    2. Buffer      — examples added; oldest evicted when full
    3. Train       — N gradient steps from a random sample of the buffer
    4. Evaluate    — every eval_every iters: current net vs best net
    5. Promote     — if win_rate ≥ win_threshold: best net ← current net
    6. Checkpoint  — every save_every iters: trainer + buffer → disk
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    book = book_path if book_path.exists() else None
    if book is None:
        print(f"[warn] opening book not found at {book_path}, skipping")

    # ------------------------------------------------------------------
    # Initialise models
    # ------------------------------------------------------------------
    net = ChessNet(num_blocks=num_blocks, num_filters=num_filters).to(device)

    best_ckpt    = checkpoints_dir / "best.pt"
    trainer_ckpt = checkpoints_dir / "trainer.pt"
    buffer_ckpt  = data_dir        / "buffer.npz"
    state_file   = checkpoints_dir / "state.pt"

    # Load best model if one exists (resume or fresh start)
    if best_ckpt.exists():
        saved = torch.load(best_ckpt, map_location=device)
        net.load_state_dict(saved["state_dict"])
        print(f"Resumed model from {best_ckpt}")

    # best_net is a frozen snapshot updated only on promotion
    best_net = ChessNet(num_blocks=num_blocks, num_filters=num_filters).to(device)
    best_net.load_state_dict(net.state_dict())

    # ------------------------------------------------------------------
    # Initialise trainer / buffer
    # ------------------------------------------------------------------
    trainer = Trainer(net, device, lr=lr, weight_decay=weight_decay,
                      batch_size=batch_size)
    if trainer_ckpt.exists():
        trainer.load_checkpoint(trainer_ckpt)
        print(f"Resumed trainer from {trainer_ckpt}")

    buffer = ReplayBuffer(maxlen=buffer_maxlen)
    if buffer_ckpt.exists():
        buffer = ReplayBuffer.load(buffer_ckpt, maxlen=buffer_maxlen)
        print(f"Resumed buffer: {buffer}")

    evaluator = Evaluator(
        device,
        num_games=eval_games,
        num_simulations=eval_sims,
        win_threshold=win_threshold,
    )

    # ------------------------------------------------------------------
    # Determine starting iteration
    # ------------------------------------------------------------------
    start_iter = 1
    if state_file.exists():
        start_iter = torch.load(state_file)["iteration"] + 1
        print(f"Resuming from iteration {start_iter}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for iteration in range(start_iter, num_iters + 1):
        print(f"\n{'='*56}")
        print(f"  Iteration {iteration}/{num_iters}")
        print(f"{'='*56}")

        # 1. Self-play ------------------------------------------------
        net.eval()
        examples = generate_games(
            net, device,
            num_games=games_per_iter,
            num_simulations=mcts_sims,
            book_path=book,
            max_moves=max_game_moves,
        )
        buffer.add(examples)
        print(f"  self-play : +{len(examples):>5} examples  {buffer}")

        # 2. Training -------------------------------------------------
        if buffer.is_ready(min_buffer):
            metrics = trainer.train(buffer, num_steps=train_steps)
            print(
                f"  train     : loss={metrics['loss']:.4f}  "
                f"pol={metrics['policy_loss']:.4f}  "
                f"val={metrics['value_loss']:.4f}"
            )

            # 3. Evaluation -------------------------------------------
            if iteration % eval_every == 0:
                promoted, stats = evaluator.evaluate(
                    net, best_net,
                    book_path=book,
                    max_moves=max_game_moves,
                    elo_log_path=elo_log_path,
                    iteration=iteration,
                )
                tag = "PROMOTED" if promoted else "not promoted"
                print(
                    f"  eval      : W={stats['wins']} D={stats['draws']} L={stats['losses']}  "
                    f"wr={stats['win_rate']:.3f}  "
                    f"elo_diff={stats['elo_diff']:+.1f}  [{tag}]"
                )

                if promoted:
                    best_net.load_state_dict(net.state_dict())
                    best_net.save(best_ckpt)
                    print(f"  best model saved → {best_ckpt}")
        else:
            print(f"  train     : waiting for buffer ({len(buffer)}/{min_buffer})")

        # 4. Periodic checkpoint (always runs, regardless of buffer state)
        if iteration % save_every == 0:
            trainer.save_checkpoint(trainer_ckpt)
            buffer.save(buffer_ckpt)
            torch.save({"iteration": iteration}, state_file)
            print(f"  checkpoint saved (iter {iteration})")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()
