---
name: Chess engine project state
description: Current implementation status and key decisions for the AlphaZero chess engine
type: project
---

All 8 modules fully implemented and tested. Training-ready on Google Colab Pro A100.

**Why:** Self-learning chess engine inspired by AlphaZero, trains via self-play with no human game data.

**How to apply:** When resuming work, everything is implemented — focus is on running training and iterating on results, not building infrastructure.

## Key decisions made
- Batched self-play: `generate_games_batched(num_games=100)` runs all games in parallel, one NN forward pass per MCTS step — ~10× speedup over sequential on A100
- `torch.compile(mode="reduce-overhead")` wraps inference net; first iteration takes ~2-3 min warmup
- Push/pop instead of `game.clone()` in MCTS simulation (~3× faster per sim)
- bfloat16 autocast for CUDA inference (A100 native)
- A100-tuned defaults: `games_per_iter=100`, `batch_size=512`
- Model: 10 residual blocks × 128 filters, ~3.6M params
- Checkpoint format is self-describing (stores num_blocks/num_filters alongside weights)

## Drive path
`/content/drive/MyDrive/chess-engine` (user uploads folder directly to MyDrive, not via Desktop sync)

## Expected training performance (A100)
~1 min/iteration → 1000 iterations in one 12h session
ELO ceiling ~1700-1900 for this architecture (10×128, 200 sims)
