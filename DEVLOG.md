# Chess Engine — Dev Log

## Training observations (iterations 1–70)

### Loss curve
| Iteration | Policy loss | Value loss | Avg moves | Self-play time |
|---|---|---|---|---|
| 1 | 6.747 | 0.165 | 145.8 | 618s |
| 32 | 2.615 | 0.042 | 57 | 348s |
| 66 | 1.950 | 0.022 | 49 | 252s |
| 71 | 1.889 | 0.021 | 47 | 224s |

- Loss is dropping consistently — no plateaus, no spikes. Training is healthy.
- Policy loss < 2.0 at iter 66 is a milestone (near-random baseline is ~8.45).
- Value loss very low (0.022) — value head understands game outcomes well.

### Game length trend
- Iter 1: 145 avg moves — random play, games drag to near the 250-move cap
- Iter 66: 49 avg moves — engine learning decisive play, normal chess length
- Shortening = positive signal. Engine is learning real patterns, not wandering.

### ELO log (iter 5–70)
- 0 wins across all 14 evaluation runs
- Mostly draws (18–20 per eval), 0–2 losses
- Win rate hovers at ~0.475–0.500 — just below the 0.55 promotion threshold
- **This is expected through ~iter 70.** Both challenger and best are similarly weak early on. Neither can convert an advantage into a checkmate.
- **Iter 80: FIRST WIN.** 1 win, 19 draws, 0 losses. Win rate 0.525, ELO diff +17.4. Not promoted (needs 0.55) but model has learned to deliver checkmate. Prediction of 75–80 was correct.
- Iter 85: back to 0 wins, 20 draws. Expected variance — model is on the edge of the threshold with only 20 eval games.
- Open question: are draws ending by move cap (250) or actual draw rules (stalemate, insufficient material)? Move-cap draws = engine hasn't learned to convert advantages. The `max_moves=100` change addresses this.

### Tail problem (long games)
- At iter 66: 99/100 games finished by outer iter ~9,500 (elapsed 225s). One game ran to outer iter 15,301 (elapsed 252s).
- The last game accounts for ~11% of wall time. GPU is nearly idle (batch size 1 instead of 100).
- Worse: positions from move-cap games are labeled as draws even when the position was decisive — corrupts value head training signal.
- **Fix:** lower `max_moves` from 250 → 100. Average game is 49 moves so this cuts off only true runaway games.

### Replay buffer
- 375,327 / 500,000 at iter 66. Will hit cap around iteration 80–90.
- After that, old random-play examples start getting evicted — training quality improves.

---

## Parameter decisions

| Parameter | Current | Decision | Reason |
|---|---|---|---|
| `max_moves` | 250 | **Change to 100** at next resume | Tail problem: GPU idle, noisy value labels |
| `mcts_sims` | 100 | Keep | Network is the bottleneck, not search depth. Revisit at iter 200–300. |
| `train_steps` | 200 | Keep | — |
| `eval_games` | 20 | Keep | — |
| Architecture | 10×128 ResNet | Keep | — |

---

## Expected milestones

| Iteration | ELO | What it plays like |
|---|---|---|
| 75–80 | ~600 | **First wins should appear in evaluation** |
| 150 | ~1000 | Basic tactics, regular wins |
| 280 | ~1200–1300 | Approaching simple checkmate patterns |
| 300 | ~1300 | Simple checkmate patterns |
| 500 | ~1500 | Solid amateur play |

Ceiling: ~1500–1700 ELO set by model size (10×128) and sim count (100).
Bumping `mcts_sims` to 200 raises ceiling to ~1700–1900 at ~2× iteration cost.

---

## Compute budget

- A100 on Colab Pro: ~5.37 units/hour
- Current pace: ~5.5 min/iter average (including eval every 5 iters)
- **$9.99 (100 units) ≈ 18.6 hours ≈ ~200 iterations**
- Starting from iter ~80: gets to **~iteration 280**

---

## Project assessment

### Strengths
- End-to-end implementation from scratch: board encoding, MCTS, neural network, training loop, evaluation pipeline
- Real performance bottlenecks found and fixed:
  - `claim_draw=True` → 4× slowdown (walked full move history at every tree node). Removed → 5 → 21 iter/s.
  - `game.clone()` replaced with push/pop → ~3× faster tree search
  - GPU→CPU transfer batched before expand loop (not inside it)
  - `torch.compile(reduce-overhead)` removed — dynamic batch size caused CUDA graph recompilation on every new size
- Self-learning with no human game data — the core AlphaZero insight
- Production-quality engineering: checkpointing, auto-resume, per-iteration CSV logging, per-phase timing

### What it demonstrates to a hiring manager
- Systems thinking (profiling, bottleneck identification, fix-and-measure)
- ML fundamentals (policy/value networks, MCTS, replay buffers, loss functions)
- Engineering discipline (checkpointing, logging, clean module boundaries)
- Shipped something non-trivial that actually runs and trains

### Weaknesses
- Architecture and algorithm are well-established — not novel research
- ELO ceiling of ~1500–1700 means it won't compete with real engines
- No UI or demo — hard to show non-technical audiences

### Recommendation
Add a short writeup covering the bottlenecks found and fixed (especially the `claim_draw` and `clone` discoveries). That's the most memorable and differentiated part of the project — most people who implement AlphaZero don't go through the profiling cycle.
