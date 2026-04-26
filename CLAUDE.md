# Chess Engine

AlphaZero-style self-play chess engine. ResNet + MCTS, trains on Google Colab Pro (A100).

## Status — paused at iteration 220

| Metric | Value |
|---|---|
| Iterations completed | 220 |
| Policy loss | 6.75 → 0.92 |
| Value loss | 0.165 → 0.003 |
| Avg game length | 145 → 46 moves |
| Real-world ELO | ~700–900 (tested vs Stockfish) |
| Replay buffer | Full at 500k examples |
| Never promoted | best.pt = bootstrap; trainer.pt = iter 220 |

Training logs: `runs/train_log.csv`, `runs/elo_log.csv`

## Resume

```bash
# Local
source .venv/bin/activate
python play.py               # CLI play (--sims 50, --black, --model)

# Colab
# Cell 1: mount Drive
# Cell 5: resume from iter 220
```

Colab path: `/content/drive/MyDrive/chess-engine`  
Drive Desktop auto-syncs checkpoints local ↔ Colab.  
`OSError: [Errno 107]` = Drive disconnected → remount Cell 1, then Cell 5.

## Architecture

**Network** (`model.py`): ResNet 10×128, ~3.6M params  
- Input: `(18, 8, 8)` — 6 piece types × 2 colors + castling + en-passant + turn  
- Policy head → 4672 logits (64×64 + underpromotions)  
- Value head → tanh scalar  
- Checkpoint: `{"num_blocks", "num_filters", "state_dict"}` — self-describing

**MCTS** (`mcts.py`): PUCT, push/pop (no clone), Dirichlet noise (α=0.3, ε=0.25) at root  
- Q stored from current player's POV, negated at parent  
- Root primed to visit_count=1 before sims

**Self-play** (`self_play.py`): 100 parallel games, one batched forward pass per outer iter  
- 4 random opening plies before MCTS (added at iter 220 for diversity)  
- temp=1.0 for first 30 half-moves, then greedy  
- GPU→CPU transfer batched before expand loop  
- Bottleneck: Python/python-chess overhead (~50–60 iter/s ceiling on A100)

**Training** (`main.py`): self-play → buffer → 200 train steps → eval every 5 iters  
- Promotion threshold: 55% win rate  
- Cosine LR 1e-3 → 1e-5, Adam, batch 512  
- No `torch.compile` — dynamic batch sizes trigger CUDA graph recompilation

## Colab parameters (notebooks/train_colab.ipynb)

```
mcts_sims=100, max_game_moves=100, train_steps=200
eval_games=20, eval_sims=100, eval_every=5
```

## Checkpoints

| File | Contents |
|---|---|
| `checkpoints/trainer.pt` | iter 220 weights + Adam state |
| `checkpoints/best.pt` | bootstrap (never promoted — see findings) |
| `checkpoints/state.pt` | `{"iteration": 220}` |
| `data/buffer.npz` | 500k replay examples |

## Key findings from 220 iterations

**Performance wins**
- `claim_draw=False` in `is_game_over()`: 5 → 21 iter/s (4× — it walked full move history at every tree node)
- Batched GPU→CPU transfer outside expand loop: meaningful speedup
- Push/pop instead of `game.clone()`: ~3× faster tree search
- `torch.compile` made it slower — removed

**Evaluation was broken**
- `temp=0` + 20 eval games = 2 unique games repeated 10 times each (deterministic play)
- All evals showed W=0 D=20 L=0 from iter 85 to 220 — promotion never triggered
- Not a training failure; eval just measured nothing useful

**Self-play ELO ≠ real ELO**
- Metrics looked good (pol loss 0.92) but engine lost to Stockfish Skill 0
- Root cause: self-play created a closed ecosystem — both sides reinforced h4/b4/f3 openings, value head never saw material imbalances
- Engine couldn't identify hanging pieces (value head uncalibrated for captures)
- After 220 iters, policy peaked so hard that temp=1 + Dirichlet noise couldn't force exploration

**What to do next (when resuming)**
- The 4 random opening plies fix is already in `self_play.py` — will take effect immediately
- Eval methodology needs fixing before promotion is meaningful (add temp > 0, or increase games to 100+)
- Real strength requires ~400–500 iters minimum; model ceiling ~1400–1500 ELO at 100 sims
- Consider bumping `mcts_sims` to 200 around iter 300 for stronger play at eval time

## Hard rules (don't break these)

- No `claim_draw=True` in `is_game_over()` — 4× slowdown
- No `logits_batch[k].cpu()` inside expand loop — batch it before
- No `torch.compile(mode="reduce-overhead")` — dynamic batch = recompile per size
- No `max_game_moves=` in `generate_games_batched` — param is `max_moves=`
- Keep Polyglot reader open from `__init__` — don't reopen in `book_move()`
- Don't install packages outside `.venv`
- Don't commit `.venv/`, `checkpoints/`, `data/`
