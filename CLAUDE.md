# Chess Engine — Claude Code Context

## Project overview
A self-learning chess engine that trains by playing against itself, inspired by AlphaZero.
The neural network uses MCTS-guided self-play to iteratively improve without any human game data.

## Current state
All modules fully implemented and tested. Training runs on Google Colab Pro (A100).

| Module | Status | Notes |
|---|---|---|
| `game.py` | ✅ complete | Board encoding, move↔index, legal-move mask, result, opening book |
| `model.py` | ✅ complete | ResNet 10×128, policy+value heads, save/load with arch config |
| `mcts.py` | ✅ complete | PUCT, Dirichlet noise, push/pop (no clone), module-level primitives |
| `self_play.py` | ✅ complete | Sequential `play_game` + batched `generate_games_batched` |
| `replay_buffer.py` | ✅ complete | Circular deque, sample, save/load (.npz) |
| `trainer.py` | ✅ complete | CE+MSE loss, cosine LR, checkpoint (model+optimizer) |
| `evaluator.py` | ✅ complete | Head-to-head eval, promotion logic, ELO log |
| `main.py` | ✅ complete | Full loop, auto-resume, torch.compile, batched self-play |
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
- Module-level primitives `_run_selection`, `_do_backup`, `_puct_select` shared by both
  the single-game `MCTS` class and `generate_games_batched`
- Root primed to `visit_count=1` before simulations so first PUCT exploration is non-zero
- Dirichlet noise (`α=0.3, ε=0.25`) added at root during self-play only

### Batched self-play (`self_play.py`)
- `generate_games_batched(num_games=100)` runs all games in parallel
- Every outer iteration: selection on all active games → leaf states stacked → **one forward pass**
- Reduces GPU round-trips from 250k × batch=1 to ~10k × batch≈100 on A100
- bfloat16 autocast on CUDA (A100 native precision, 2× throughput over fp32)
- Temperature schedule: `τ=1.0` for first 30 half-moves, `τ=0.0` after
- Book moves advanced without MCTS and excluded from the training buffer

### Training loop (`main.py`)
- `torch.compile(net, mode="reduce-overhead")` wraps inference net; shares weights with training net
- Compile warmup ~2–3 min on first iteration, then negligible
- Per-iteration: self-play → buffer → train (500 steps) → eval every 5 iters
- Promotion: new net replaces best net if `(wins + 0.5×draws) / total ≥ 0.55`
- ELO diff logged: `400 × log10(win_rate / (1 − win_rate))`

## A100 performance (Colab Pro)

| Phase | Unbatched (old) | Batched (current) |
|---|---|---|
| Self-play (100 games) | ~20 min | ~25 s |
| Training (500 steps, bs=512) | ~20 s | ~15 s |
| Eval amortised (÷5) | ~3 min | ~25 s |
| **Total per iteration** | **~25 min** | **~1 min** |

1000 iterations ≈ one 12-hour Colab session.

### ELO milestones (absolute vs fixed benchmark)
| Iterations | Wall time | ELO |
|---|---|---|
| 50 | ~1 h | ~800 |
| 150 | ~3 h | ~1100 |
| 300 | ~5 h | ~1400 |
| 1000 | ~17 h | ~1700–1900 |

Ceiling ~1700–1900 set by model size (10×128) and sim count (200). Not a training-time limit.

## Key implementation details

| Detail | Value |
|---|---|
| Board encoding | 18 planes, (18, 8, 8) float32, always from current player's POV |
| Move space | 4672 — 64×64 from/to pairs + 64×9 underpromotions |
| Policy loss | Cross-entropy with soft MCTS targets: `−Σ π·log softmax(logits)` |
| Value loss | MSE: `(v_pred − z)²` |
| Optimizer | Adam, lr=1e-3, cosine decay to 1e-5 per training call |
| Replay buffer | 500k examples max, deque eviction |
| Evaluation | 40 games, half as White / half as Black, temp=0, no noise |

## Checkpoints

| File | Contents |
|---|---|
| `checkpoints/best.pt` | Best model weights + arch config (`num_blocks`, `num_filters`) |
| `checkpoints/trainer.pt` | Model weights + Adam optimizer state |
| `checkpoints/state.pt` | `{"iteration": N}` — resume marker |
| `data/buffer.npz` | Full replay buffer (states, policies, values) |
| `runs/elo_log.csv` | `iteration, wins, draws, losses, win_rate, elo_diff, promoted` |

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
│   ├── self_play.py           ← sequential play_game + generate_games_batched
│   ├── replay_buffer.py       ← circular buffer, sample, save/load
│   ├── trainer.py             ← CE+MSE loss, cosine LR, checkpointing
│   ├── evaluator.py           ← head-to-head eval, promotion, ELO log
│   └── main.py                ← orchestration loop, torch.compile, auto-resume
├── checkpoints/               ← model weights (synced to Drive)
├── data/                      ← replay buffer snapshots (synced to Drive)
├── books/
│   └── gm2001.bin             ← Polyglot opening book
├── runs/
│   └── elo_log.csv            ← ELO log (header pre-seeded)
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

## What NOT to do
- Don't install packages outside the venv
- Don't commit `.venv/`, `checkpoints/`, or `data/` to git (see .gitignore)
- Don't hardcode local paths — use `pathlib.Path` relative to project root
- Don't run full training locally — use Colab for anything GPU-heavy
- Don't call `generate_games_batched` with `max_game_moves=` — the parameter is `max_moves=`
