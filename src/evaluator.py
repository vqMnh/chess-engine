"""Evaluator: pits new model against old; promotes new model if win rate >= 55%."""

import math
import numpy as np
import torch
import torch.nn as nn
import chess
from pathlib import Path
from tqdm import tqdm

from game import ChessGame
from mcts import MCTS


def _play_match(
    white_model: nn.Module,
    black_model: nn.Module,
    device: torch.device,
    num_simulations: int,
    c_puct: float,
    book_path: Path | None,
    max_moves: int,
) -> float:
    """Play one game. Returns result from White's perspective: +1/0/−1."""
    game = ChessGame(book_path=book_path)
    white_mcts = MCTS(white_model, device, num_simulations=num_simulations, c_puct=c_puct)
    black_mcts = MCTS(black_model, device, num_simulations=num_simulations, c_puct=c_puct)

    ply = 0
    while not game.is_game_over() and ply < max_moves:
        book_move = game.book_move()
        if book_move is not None:
            game.push(book_move)
            ply += 1
            continue

        mcts = white_mcts if game.turn == chess.WHITE else black_mcts
        # temperature=0 → deterministic one-hot; no Dirichlet noise for fair evaluation
        policy = mcts.get_policy(game, temperature=0.0, add_noise=False)
        game.push_index(int(np.argmax(policy)))
        ply += 1

    result = game.result()
    return result if result is not None else 0.0


class Evaluator:
    """Runs head-to-head games between new and old model to decide promotion.

    New model plays White for half the games and Black for the other half to
    remove first-move advantage bias.  Promotion requires win_rate >= threshold
    where win_rate = (wins + 0.5*draws) / total_games.
    """

    def __init__(
        self,
        device: torch.device,
        num_games: int = 40,
        num_simulations: int = 200,
        win_threshold: float = 0.55,
        c_puct: float = 1.5,
    ):
        self.device = device
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.win_threshold = win_threshold
        self.c_puct = c_puct

    def evaluate(
        self,
        new_model: nn.Module,
        old_model: nn.Module,
        book_path: Path | None = None,
        max_moves: int = 512,
        elo_log_path: Path | None = None,
        iteration: int = 0,
    ) -> tuple[bool, dict]:
        """Pit new_model against old_model and decide whether to promote.

        Args:
            new_model:    Candidate model (already in eval mode recommended).
            old_model:    Current best model.
            book_path:    Optional Polyglot book for both sides.
            max_moves:    Hard cap per game; counted as draw if reached.
            elo_log_path: Path to the ELO CSV log (appended, not overwritten).
            iteration:    Training iteration number for the log row.

        Returns:
            (promoted, stats) where stats contains wins/draws/losses/win_rate/elo_diff.
        """
        new_model.eval()
        old_model.eval()

        wins = draws = losses = 0
        half = self.num_games // 2

        kwargs = dict(
            device=self.device,
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            book_path=book_path,
            max_moves=max_moves,
        )

        for _ in tqdm(range(half), desc="eval (new=W)", leave=False):
            r = _play_match(new_model, old_model, **kwargs)
            if   r > 0:  wins   += 1
            elif r == 0: draws  += 1
            else:        losses += 1

        for _ in tqdm(range(self.num_games - half), desc="eval (new=B)", leave=False):
            r = _play_match(old_model, new_model, **kwargs)
            if   r < 0:  wins   += 1   # new_model is Black; Black win → negative r
            elif r == 0: draws  += 1
            else:        losses += 1

        total    = wins + draws + losses
        win_rate = (wins + 0.5 * draws) / max(total, 1)

        # ELO difference relative to old model
        if 0 < win_rate < 1:
            elo_diff = 400 * math.log10(win_rate / (1 - win_rate))
        elif win_rate >= 1:
            elo_diff = 800.0   # cap at +800 (never loses or draws)
        else:
            elo_diff = -800.0  # cap at -800

        promoted = win_rate >= self.win_threshold

        stats: dict = {
            "wins":     wins,
            "draws":    draws,
            "losses":   losses,
            "win_rate": win_rate,
            "elo_diff": elo_diff,
        }

        if elo_log_path is not None:
            self._log(elo_log_path, iteration, stats, promoted)

        return promoted, stats

    # ------------------------------------------------------------------
    # ELO logging
    # ------------------------------------------------------------------

    def _log(self, path: Path, iteration: int, stats: dict, promoted: bool) -> None:
        """Append one row to the CSV log; write header if the file is empty."""
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with open(path, "a") as f:
            if write_header:
                f.write("iteration,wins,draws,losses,win_rate,elo_diff,promoted\n")
            f.write(
                f"{iteration},"
                f"{stats['wins']},{stats['draws']},{stats['losses']},"
                f"{stats['win_rate']:.4f},{stats['elo_diff']:.1f},"
                f"{int(promoted)}\n"
            )
