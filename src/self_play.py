"""Self-play loop: generates (state, π, z) training tuples via MCTS-guided games."""

import numpy as np
import torch
import chess
from pathlib import Path

from game import ChessGame, POLICY_SIZE
from mcts import MCTS

# Type alias for a single training example
SelfPlayExample = tuple[np.ndarray, np.ndarray, float]

# AlphaZero temperature schedule: exploratory for first N half-moves, greedy after
_TEMP_THRESHOLD = 30


def play_game(
    model: torch.nn.Module,
    device: torch.device,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    book_path: Path | None = None,
    max_moves: int = 512,
) -> list[SelfPlayExample]:
    """Play one self-play game and return (state, policy, value) training examples.

    Book moves are played to reach interesting positions quickly but are not
    added to the training buffer (no MCTS policy to supervise against).

    Args:
        model:           Neural network in eval mode.
        device:          Torch device for inference.
        num_simulations: MCTS simulations per move.
        c_puct:          PUCT exploration constant.
        book_path:       Optional Polyglot opening book.
        max_moves:       Hard cap on game length; result treated as draw if hit.

    Returns:
        List of (state, π, z) where:
          state  — float32 array (NUM_PLANES, 8, 8), current-player perspective
          π      — float32 array (POLICY_SIZE,), MCTS visit-count distribution
          z      — float, game outcome from the current player's perspective (+1/0/−1)
    """
    game = ChessGame(book_path=book_path)
    mcts = MCTS(model, device, num_simulations=num_simulations, c_puct=c_puct)

    # Store (state, policy, turn) — value filled in once the game ends
    buffer: list[tuple[np.ndarray, np.ndarray, chess.Color]] = []

    ply = 0
    while not game.is_game_over() and ply < max_moves:
        # Opening book: play the move but skip recording (no MCTS policy)
        book_move = game.book_move()
        if book_move is not None:
            game.push(book_move)
            ply += 1
            continue

        temperature = 1.0 if ply < _TEMP_THRESHOLD else 0.0

        state = game.encode()
        policy = mcts.get_policy(game, temperature=temperature, add_noise=True)

        buffer.append((state, policy, game.turn))

        # Sample the move proportional to policy
        legal = game.legal_move_indices()
        probs = np.array([policy[i] for i in legal], dtype=np.float64)
        probs /= probs.sum()
        chosen = int(np.random.choice(legal, p=probs))
        game.push_index(chosen)
        ply += 1

    result = game.result()
    if result is None:
        result = 0.0  # max_moves reached → draw

    examples: list[SelfPlayExample] = []
    for state, policy, turn in buffer:
        # z is from the perspective of whoever was to move at that state
        z = float(result) if turn == chess.WHITE else -float(result)
        examples.append((state, policy, z))

    return examples


def generate_games(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    book_path: Path | None = None,
    max_moves: int = 512,
) -> list[SelfPlayExample]:
    """Play num_games self-play games and return all training examples."""
    examples: list[SelfPlayExample] = []
    for _ in range(num_games):
        examples.extend(
            play_game(
                model, device,
                num_simulations=num_simulations,
                c_puct=c_puct,
                book_path=book_path,
                max_moves=max_moves,
            )
        )
    return examples
