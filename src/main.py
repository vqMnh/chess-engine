"""Entry point: orchestrates the self-play → train → evaluate loop."""

import torch
from pathlib import Path

from model import ChessNet
from self_play import generate_games_batched
from replay_buffer import ReplayBuffer
from trainer import Trainer
from evaluator import Evaluator


def train(
    # --- Paths ---
    checkpoints_dir: Path  = Path("checkpoints"),
    data_dir:        Path  = Path("data"),
    book_path:       Path  = Path("books/gm2001.bin"),
    elo_log_path:    Path  = Path("runs/elo_log.csv"),
    # --- Architecture ---
    num_blocks:  int = 10,
    num_filters: int = 128,
    # --- Self-play ---
    # ↑ 100 parallel games keeps A100 batch ≈ 80-100; raise to 200 for even
    #   better utilisation (costs more RAM but A100 has 40 GB to spare).
    games_per_iter:  int   = 100,
    mcts_sims:       int   = 200,
    max_game_moves:  int   = 512,
    # --- Training ---
    train_steps:   int   = 500,
    batch_size:    int   = 512,    # A100 can saturate larger batches
    lr:            float = 1e-3,
    weight_decay:  float = 1e-4,
    buffer_maxlen: int   = 500_000,
    min_buffer:    int   = 1_000,
    # --- Evaluation ---
    eval_every:    int   = 5,
    eval_games:    int   = 40,
    eval_sims:     int   = 200,
    win_threshold: float = 0.55,
    # --- Loop ---
    num_iters:  int = 1_000,
    save_every: int = 10,
) -> None:
    """Run the AlphaZero self-play → train → evaluate loop.

    All ``games_per_iter`` games run in parallel: every MCTS simulation step
    batches their leaf evaluations into one forward pass.  On an A100 with
    games_per_iter=100 this drives GPU utilisation from <5 % (sequential) to
    >60 %, cutting iteration time from ~7 min to ~45 s.

    Loop per iteration
    ──────────────────
    1. Batched self-play  →  (state, π, z) examples
    2. Replay buffer      →  oldest examples evicted when full
    3. Train              →  N gradient steps
    4. Evaluate           →  every eval_every iters: current vs best net
    5. Promote            →  win_rate ≥ win_threshold: best net ← current net
    6. Checkpoint         →  every save_every iters: weights + buffer → disk
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    book = book_path if book_path.exists() else None
    if book is None:
        print(f"[warn] opening book not found at {book_path}, skipping")

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    net = ChessNet(num_blocks=num_blocks, num_filters=num_filters).to(device)

    best_ckpt    = checkpoints_dir / "best.pt"
    trainer_ckpt = checkpoints_dir / "trainer.pt"
    buffer_ckpt  = data_dir        / "buffer.npz"
    state_file   = checkpoints_dir / "state.pt"

    if best_ckpt.exists():
        saved = torch.load(best_ckpt, map_location=device)
        net.load_state_dict(saved["state_dict"])
        print(f"Resumed model from {best_ckpt}")

    best_net = ChessNet(num_blocks=num_blocks, num_filters=num_filters).to(device)
    best_net.load_state_dict(net.state_dict())

    # Compile for inference: fuses ops and reduces CUDA kernel-launch overhead.
    # Shares weights with net — training updates are reflected automatically.
    if device.type == "cuda":
        inference_net = torch.compile(net, mode="reduce-overhead")
        print("torch.compile enabled for self-play inference")
    else:
        inference_net = net

    # ------------------------------------------------------------------
    # Trainer / buffer
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

    evaluator = Evaluator(device, num_games=eval_games,
                          num_simulations=eval_sims, win_threshold=win_threshold)

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

        # 1. Batched self-play ----------------------------------------
        net.eval()
        examples = generate_games_batched(
            inference_net, device,
            num_games        = games_per_iter,
            num_simulations  = mcts_sims,
            book_path        = book,
            max_moves        = max_game_moves,
        )
        buffer.add(examples)
        print(f"  self-play : +{len(examples):>5} examples  {buffer}")

        # 2. Training -------------------------------------------------
        if buffer.is_ready(min_buffer):
            net.train()
            metrics = trainer.train(buffer, num_steps=train_steps)
            print(
                f"  train     : loss={metrics['loss']:.4f}  "
                f"pol={metrics['policy_loss']:.4f}  "
                f"val={metrics['value_loss']:.4f}"
            )

            # 3. Evaluation -------------------------------------------
            if iteration % eval_every == 0:
                net.eval()
                promoted, stats = evaluator.evaluate(
                    net, best_net,
                    book_path  = book,
                    max_moves  = max_game_moves,
                    elo_log_path = elo_log_path,
                    iteration  = iteration,
                )
                tag = "PROMOTED" if promoted else "not promoted"
                print(
                    f"  eval      : W={stats['wins']} D={stats['draws']} "
                    f"L={stats['losses']}  wr={stats['win_rate']:.3f}  "
                    f"elo_diff={stats['elo_diff']:+.1f}  [{tag}]"
                )
                if promoted:
                    best_net.load_state_dict(net.state_dict())
                    best_net.save(best_ckpt)
                    print(f"  best model saved → {best_ckpt}")
        else:
            print(f"  train     : waiting for buffer ({len(buffer)}/{min_buffer})")

        # 4. Periodic checkpoint (always runs) ------------------------
        if iteration % save_every == 0:
            trainer.save_checkpoint(trainer_ckpt)
            buffer.save(buffer_ckpt)
            torch.save({"iteration": iteration}, state_file)
            print(f"  checkpoint saved (iter {iteration})")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()
