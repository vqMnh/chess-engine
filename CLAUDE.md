# Chess Engine — Claude Code Context

## Project overview
A self-learning chess engine that trains by playing against itself, inspired by AlphaZero.
The neural network uses MCTS-guided self-play to iteratively improve without any human game data.

## Current state
- Scaffold complete; venv initialized with all deps installed and verified (`chess`, `torch`, `torchvision`, `numpy`, `tqdm`)
- `game.py` fully implemented and tested — board encoding, move index round-trips, legal-move mask, result, opening book lookup all pass
- All other `src/` modules are stubs (docstring + imports only) — ready for implementation
- Colab notebook has 5 empty cells: Mount Drive, Install deps, Import, Train, Resume
- `checkpoints/`, `data/`, `books/`, `runs/` directories created; `runs/elo_log.csv` seeded with header
- Opening book present at `books/gm2001.bin`

## Architecture
- **MCTS search** — guides move selection during self-play using UCB/PUCT
- **Neural network** — residual CNN with two heads:
  - Policy head → probability distribution over all legal moves
  - Value head → scalar win probability in [−1, 1]
- **Self-play loop** — engine plays itself, generating (state, π, z) training tuples
- **Replay buffer** — stores recent game data, sampled randomly for training
- **Evaluation** — new model vs. old model; replaces old if win rate ≥ 55%

### Planned improvements
- **Opening book** — preload a small Polyglot `.bin` opening book via `chess.polyglot`. During self-play, follow book moves for the first N plies before handing off to MCTS. Prevents completely random early games and speeds up learning of middlegame patterns.
- **Batched MCTS leaf evaluation** — instead of calling the neural net once per leaf node, queue all pending leaf nodes and evaluate them in a single batched forward pass. This is the single biggest speed lever in the whole pipeline — can give 10–50× throughput improvement on GPU.
- **ELO tracking** — after each evaluation round, compute an ELO estimate from the win rate using the standard formula: `ELO_diff = 400 * log10(W / (1 - W))`. Log it to `runs/elo_log.csv` so progress is visible over iterations.

## Environment setup
This project uses a dedicated virtual environment. **Never install packages globally.**

```bash
# Create venv (first time only)
python -m venv .venv

# Activate (Mac/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install deps
pip install -r requirements.txt
```

## Project structure
```
chess-engine/
├── CLAUDE.md                  ← you are here
├── requirements.txt
├── .venv/                     ← local venv, never committed
├── .gitignore
├── src/
│   ├── game.py                ← chess environment (python-chess wrapper)
│   ├── model.py               ← ResNet policy + value network (PyTorch)
│   ├── mcts.py                ← Monte Carlo Tree Search with PUCT
│   ├── self_play.py           ← self-play game generation
│   ├── replay_buffer.py       ← experience storage and sampling
│   ├── trainer.py             ← training loop and loss functions
│   └── evaluator.py           ← new vs. old model evaluation
├── checkpoints/               ← saved model weights (synced to Drive)
├── data/                      ← replay buffer snapshots (synced to Drive)
├── books/
│   └── .gitkeep               ← drop a Polyglot .bin opening book here
├── runs/
│   └── elo_log.csv            ← ELO estimates logged after each evaluation round
└── notebooks/
    └── train_colab.ipynb      ← Colab training entry point
```

## Compute split
| Task | Where |
|---|---|
| Code editing | Local machine |
| Local testing / debugging | Local machine (venv) |
| Full training runs | Google Colab Pro (GPU) |
| Checkpoint storage | Google Drive (via Drive Desktop sync) |

## Google Drive / Colab workflow
- **Drive Desktop** syncs the local project folder to `My Drive/chess-engine/` automatically
- In Colab, mount Drive and import directly from the synced folder:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  import sys
  sys.path.insert(0, '/content/drive/MyDrive/chess-engine/src')
  ```
- Checkpoints saved by Colab land in `checkpoints/` and sync back to local automatically
- No manual uploading or downloading needed

## Key implementation details
- Board is encoded as 8×8×N binary planes (piece types × colors + metadata like castling rights, en passant)
- Move space: 4096+ dimensional policy vector (all from/to square pairs + underpromotions)
- Training loss: `MSE(value_pred, z) + CrossEntropy(policy_pred, π_mcts)`
- Optimizer: Adam, lr ≈ 0.001 with cosine decay
- MCTS uses PUCT formula for node selection: `U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`

## Dependencies
See `requirements.txt`. Core deps:
- `python-chess` — chess rules, legal move generation, FEN/PGN
- `torch` + `torchvision` — neural network
- `numpy` — board representation and MCTS arrays
- `tqdm` — training progress bars

## What NOT to do
- Don't install packages outside the venv
- Don't commit `.venv/`, `checkpoints/`, or `data/` to git (see .gitignore)
- Don't hardcode local paths — use `pathlib.Path` relative to project root
- Don't run full training locally — use Colab for anything GPU-heavy
