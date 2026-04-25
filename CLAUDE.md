# Chess Engine — Claude Code Context

## Project overview
A self-learning chess engine that trains by playing against itself, inspired by AlphaZero.
The neural network uses MCTS-guided self-play to iteratively improve without any human game data.

## Current state
All modules fully implemented and tested. Training runs on Google Colab Pro (A100).

| Module | Status | Notes |
|---|---|---|
| `game.py` | ✅ complete | Board encoding, move↔index, legal-move mask, result, opening book (reader kept open per game, no claim_draw) |
| `model.py` | ✅ complete | ResNet 10×128, policy+value heads, save/load with arch config |
| `mcts.py` | ✅ complete | PUCT, Dirichlet noise, push/pop (no clone), module-level primitives |
| `self_play.py` | ✅ complete | Batched self-play — set-based active tracking, batched GPU→CPU, single legal-moves call, iter/s logging |
| `replay_buffer.py` | ✅ complete | Circular deque, sample, save/load (.npz) |
| `trainer.py` | ✅ complete | CE+MSE loss, cosine LR, checkpoint (model+optimizer) |
| `evaluator.py` | ✅ complete | Head-to-head eval, promotion logic, ELO log |
| `main.py` | ✅ complete | Full loop, auto-resume, per-phase timing, training banner, per-iteration CSV log |
| `train_colab.ipynb` | ✅ complete | 5 cells: Mount Drive, Deps, Import, Train, Resume |

Colab Drive path: `/content/drive/MyDrive/chess-engine`

## Architecture

### Neural network (`model.py`)
- **Input**: `(18, 8, 8)` float32 — 6 piece types × 2 colors + 4 castling + 1 en-passant + 1 turn flag
- **Stem**: Conv2d(18→128, 3×3) + BN + ReLU
- **Tower**: 10 residual blocks — each Conv(128→128, 3×3) + BN + ReLU × 2 with skip connection
- **Policy head**: Conv(128→2, 1×1) + BN + ReLU → flatten → Linear(128→4672) — raw logits
- **Value head**: Conv(128→1, 1×1) + BN + ReLU → flatten → Linear(64→256) → ReLU → Linear(256→1) → Tanh
- **Parameters**: ~3.6M
- **Checkpoint format**: `{"num_blocks", "num_filters", "state_dict"}` — architecture is self-describing

### MCTS (`mcts.py`)
- PUCT formula: `score(s,a) = -Q(child) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))`
- `Q` stored from each node's own player's perspective; negated when selecting from parent
- **No `game.clone()`** — selection uses push/pop on the original game object (~3× faster)
- Module-level primitives `_run_selection`, `_do_backup`, `_puct_select` shared by
  the `MCTS` class (used by `Evaluator`) and `generate_games_batched`
- Root primed to `visit_count=1` before simulations so first PUCT exploration is non-zero
- Dirichlet noise (`α=0.3, ε=0.25`) added at root during self-play only

### Opening book (`game.py`)
- `ChessGame.__init__` opens the Polyglot reader once and stores it in `self._book_reader`
- `book_move()` uses the stored reader directly — no file I/O per call
- On the first position not in the book, `_book_exhausted` is set and the reader is closed
- `is_game_over()` uses `claim_draw=False` — only checks checkmate/stalemate (O(1))
  - Draws by repetition/50-move are handled implicitly by the `max_moves` cap
  - Using `claim_draw=True` was the second major bottleneck (walked move history at every tree level): removing it gave a **4× speedup** (5 → 21 iter/s)

### Batched self-play (`self_play.py`)
- `generate_games_batched(num_games=100)` runs all games in parallel; returns `(examples, avg_moves)`
- Every outer iteration: selection on all active games → leaf states stacked → **one forward pass**
- `active` games tracked as a `set` for O(1) removal
- GPU→CPU transfer batched once before the expand loop (`logits_batch.float().cpu().numpy()`)
- `legal_move_indices()` called once per expansion, reused for mask, Dirichlet noise, and node children
- Logs every 500 outer iterations: active count, iter/s, elapsed; summary line on completion
- **Real bottleneck**: Python MCTS overhead (python-chess `push`/`pop` per tree level). GPU is ~1% of wall time. Ceiling is ~50–60 iter/s with python-chess regardless of GPU.

### Training loop (`main.py`)
- **No `torch.compile`** — `reduce-overhead` mode uses CUDA Graphs which require fixed input shapes; our dynamic batch size (shrinks as games finish) triggered recompilation on every new size, adding overhead instead of saving it
- Prints a training banner at startup (device, all params) and session-elapsed time per iteration
- Per-phase timing printed: `[sp_secs]`, `[tr_secs]`, `[ev_secs]`
- Per-iteration CSV log appended to `runs/train_log.csv`: `iteration, sp_time_s, avg_moves, examples, train_loss, pol_loss, val_loss, iter_time_s`
- Per-iteration: self-play → buffer → train (200 steps) → eval every 5 iters
- Promotion: new net replaces best net if `(wins + 0.5×draws) / total ≥ 0.55`
- ELO diff logged: `400 × log10(win_rate / (1 − win_rate))`

## A100 performance (Colab Pro) — measured

Current parameters: `mcts_sims=100`, `max_game_moves=250`, `train_steps=200`, `eval_games=20`

| Phase | Time |
|---|---|
| Self-play iter 1 (random network, avg 134 moves/game) | ~607s (~10 min) |
| Self-play later iters (network improves, games shorten) | ~3–6 min (estimate) |
| Training (200 steps, bs=512) | ~6s |
| Eval amortised (÷5, 20 games, 100 sims) | ~TBD |
| **Total iter 1** | **~613s** |

**iter/s behaviour**: starts at ~21 iter/s (100 active games), accelerates to ~160 iter/s as games finish (fewer active games = less Python per outer iter). Overall average for iter 1: **39.6 iter/s**, 24,001 outer iterations.

Colab Pro compute: ~5.37 units/hour on A100, ~62 units/month → ~11.5 hours/month → ~70 iterations/month at current speed.

### ELO milestones (100 sims/move)
| Iterations | ELO | What it plays like |
|---|---|---|
| 50 | ~700 | Stops blundering randomly |
| 150 | ~1000 | Basic tactics |
| 300 | ~1300 | Simple checkmate patterns |
| 500 | ~1400–1500 | Solid amateur play |
| 1000 | ~1500–1700 | Approaching model ceiling |

Ceiling ~1500–1700 set by model size (10×128) and sim count (100). Increasing `mcts_sims` to 200 raises ceiling to ~1700–1900 at ~2× iteration cost.

## Key implementation details

| Detail | Value |
|---|---|
| Board encoding | 18 planes, (18, 8, 8) float32, always from current player's POV |
| Move space | 4672 — 64×64 from/to pairs + 64×9 underpromotions |
| Policy loss | Cross-entropy with soft MCTS targets: `−Σ π·log softmax(logits)` |
| Value loss | MSE: `(v_pred − z)²` |
| Optimizer | Adam, lr=1e-3, cosine decay to 1e-5 per training call |
| Replay buffer | 500k examples max, deque eviction |
| Evaluation | 20 games, half as White / half as Black, temp=0, no noise |

## Checkpoints

| File | Contents |
|---|---|
| `checkpoints/best.pt` | Best model weights + arch config (`num_blocks`, `num_filters`) |
| `checkpoints/trainer.pt` | Model weights + Adam optimizer state |
| `checkpoints/state.pt` | `{"iteration": N}` — resume marker |
| `data/buffer.npz` | Full replay buffer (states, policies, values) |
| `runs/elo_log.csv` | `iteration, wins, draws, losses, win_rate, elo_diff, promoted` |
| `runs/train_log.csv` | `iteration, sp_time_s, avg_moves, examples, train_loss, pol_loss, val_loss, iter_time_s` |

## Environment setup
This project uses a dedicated virtual environment. **Never install packages globally.**

```bash
# Create venv (first time only)
python -m venv .venv

# Activate (Mac/Linux)
source .venv/bin/activate

# Install deps
pip install -r requirements.txt
```

## Project structure
```
chess-engine/
├── CLAUDE.md
├── requirements.txt
├── .venv/                     ← local venv, never committed
├── .gitignore
├── src/
│   ├── game.py                ← chess environment (python-chess wrapper)
│   ├── model.py               ← ResNet policy + value network
│   ├── mcts.py                ← MCTS with PUCT + module-level primitives
│   ├── self_play.py           ← generate_games_batched (GPU batched self-play)
│   ├── replay_buffer.py       ← circular buffer, sample, save/load
│   ├── trainer.py             ← CE+MSE loss, cosine LR, checkpointing
│   ├── evaluator.py           ← head-to-head eval, promotion, ELO log
│   └── main.py                ← orchestration loop, per-iteration CSV log, auto-resume
├── checkpoints/               ← model weights (synced to Drive)
├── data/                      ← replay buffer snapshots (synced to Drive)
├── books/
│   └── gm2001.bin             ← Polyglot opening book
├── runs/
│   ├── elo_log.csv            ← eval results (written every eval_every iters)
│   └── train_log.csv          ← per-iteration metrics (written every iteration)
└── notebooks/
    └── train_colab.ipynb      ← 5-cell Colab entry point
```

## Compute split
| Task | Where |
|---|---|
| Code editing | Local machine |
| Local testing / debugging | Local machine (venv) |
| Full training runs | Google Colab Pro — A100-SXM4-40GB |
| Checkpoint storage | Google Drive (Drive Desktop auto-sync) |

## Google Drive / Colab workflow
- **Drive Desktop** syncs the local project folder automatically
- Colab sees it at `/content/drive/MyDrive/chess-engine`
- Checkpoints written by Colab sync back to local automatically — no manual steps
- If Drive disconnects mid-session: remount (Cell 1) then resume (Cell 5) if past iter 10, else fresh start (Cell 4)
- `OSError: [Errno 107] Transport endpoint is not connected` = Drive FUSE disconnected; remount and retry

## What NOT to do
- Don't install packages outside the venv
- Don't commit `.venv/`, `checkpoints/`, or `data/` to git (see .gitignore)
- Don't hardcode local paths — use `pathlib.Path` relative to project root
- Don't run full training locally — use Colab for anything GPU-heavy
- Don't call `generate_games_batched` with `max_game_moves=` — the parameter is `max_moves=`
- Don't reopen the Polyglot reader inside `book_move()` — it must stay open from `__init__`
- Don't use `torch.compile(mode="reduce-overhead")` — dynamic batch sizes cause CUDA graph recompilation per new size, adding overhead; no compile is faster here
- Don't use `claim_draw=True` in `is_game_over()` — walks full move history at every tree node, was a 4× slowdown
- Don't move `logits_batch[k].cpu()` inside the expand loop — batch the GPU→CPU transfer before the loop
