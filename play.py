#!/usr/bin/env python3
"""
Play against the chess engine in the terminal.

Usage:
    python play.py              # you play White
    python play.py --black      # you play Black
    python play.py --sims 100   # stronger engine (slower, ~30s/move on CPU)
    python play.py --sims 20    # faster engine (weaker,  ~5s/move on CPU)
"""

import sys, argparse, time
from pathlib import Path

import torch
import chess
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from model import ChessNet
from mcts import MCTS
from game import ChessGame, index_to_move

PIECES = {
    (chess.PAWN,   chess.WHITE): "♙", (chess.KNIGHT, chess.WHITE): "♘",
    (chess.BISHOP, chess.WHITE): "♗", (chess.ROOK,   chess.WHITE): "♖",
    (chess.QUEEN,  chess.WHITE): "♕", (chess.KING,   chess.WHITE): "♔",
    (chess.PAWN,   chess.BLACK): "♟", (chess.KNIGHT, chess.BLACK): "♞",
    (chess.BISHOP, chess.BLACK): "♝", (chess.ROOK,   chess.BLACK): "♜",
    (chess.QUEEN,  chess.BLACK): "♛", (chess.KING,   chess.BLACK): "♚",
}


def render_board(board: chess.Board, flip: bool = False) -> str:
    ranks = list(range(8))         if flip else list(range(7, -1, -1))
    files = list(range(7, -1, -1)) if flip else list(range(8))
    # Each square = 2 chars (piece + space). Header must match: 1 letter + 1 space.
    col_labels = "     " + " ".join("abcdefgh"[f] for f in files)
    border     = "   +" + "--" * 8 + "-+"

    lines = ["", col_labels, border]
    for rank in ranks:
        row = f" {rank+1} | "
        for f in files:
            sq = chess.square(f, rank)
            p  = board.piece_at(sq)
            row += (PIECES[(p.piece_type, p.color)] if p else "·") + " "
        row += f"| {rank+1}"
        lines.append(row)
    lines += [border, col_labels, ""]
    return "\n".join(lines)


def get_player_move(board: chess.Board) -> chess.Move | None:
    """Returns a legal move, or None if the player resigns."""
    while True:
        try:
            raw = input("  Your move: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if raw.lower() in ("q", "quit", "resign", "exit"):
            return None

        if raw.lower() == "help":
            print("  SAN examples : e4  Nf3  Bxc6  O-O  Qxd5+")
            print("  UCI examples : e2e4  g1f3  e1g1 (castling)")
            print("  Commands     : resign / quit")
            continue

        # Try SAN first (more natural for a real player)
        try:
            return board.parse_san(raw)
        except Exception:
            pass

        # Try UCI
        try:
            move = chess.Move.from_uci(raw)
            if move in board.legal_moves:
                return move
        except Exception:
            pass

        print(f"  Not a legal move: '{raw}'  (type 'help' for format)")


def play(model_path: Path, player_color: chess.Color, sims: int, device: torch.device) -> None:
    # ── load model ────────────────────────────────────────────────────────
    data = torch.load(model_path, map_location=device)
    key  = "model_state" if "model_state" in data else "state_dict"
    net  = ChessNet(num_blocks=data["num_blocks"], num_filters=data["num_filters"]).to(device)
    net.load_state_dict(data[key])
    net.eval()
    mcts = MCTS(net, device, num_simulations=sims, c_puct=1.5)

    flip       = (player_color == chess.BLACK)
    game       = ChessGame()   # no opening book so human picks their own opening
    san_history: list[str] = []

    side = "White" if player_color == chess.WHITE else "Black"
    print(f"\n{'─'*48}")
    print(f"  You: {side}   Engine sims: {sims}   Device: {device}")
    print(f"  Moves: SAN (e4, Nf3, O-O) or UCI (e2e4)")
    print(f"  Type  'resign'  to end the game")
    print(f"{'─'*48}")

    # ── main game loop ────────────────────────────────────────────────────
    while not game.board.is_game_over() and len(san_history) < 400:
        print(render_board(game.board, flip=flip))

        turn  = "White" if game.board.turn == chess.WHITE else "Black"
        check = "  ★ CHECK ★" if game.board.is_check() else ""
        last  = f"   last: {san_history[-1]}" if san_history else ""
        move_no = game.board.fullmove_number
        print(f"  Move {move_no}  ({turn} to play){check}{last}\n")

        if game.board.turn == player_color:
            # ── human move ────────────────────────────────────────────────
            move = get_player_move(game.board)
            if move is None:
                print("\n  You resigned. Good game!")
                return
            san = game.board.san(move)
            game.push(move)
            san_history.append(san)
            print()

        else:
            # ── engine move ───────────────────────────────────────────────
            print("  Engine thinking...", end="", flush=True)
            t0     = time.perf_counter()
            policy = mcts.get_policy(game, temperature=0.0, add_noise=False)
            idx    = int(np.argmax(policy))
            secs   = time.perf_counter() - t0

            engine_move = index_to_move(idx, game.board)
            san         = game.board.san(engine_move)
            game.push_index(idx)
            san_history.append(san)
            print(f"\r  Engine plays: {san:<8}  ({secs:.1f}s)\n")

    # ── final board + result ──────────────────────────────────────────────
    print(render_board(game.board, flip=flip))

    outcome = game.board.outcome()
    if outcome is None:
        print("  Game drawn (move limit).")
    elif outcome.winner is None:
        print(f"  Draw — {outcome.termination.name.replace('_', ' ').lower()}.")
    elif outcome.winner == player_color:
        print("  You won! Well played.")
    else:
        print("  Engine wins.")

    if san_history:
        moves_str = "  ".join(
            f"{i//2+1}.{m}" if i % 2 == 0 else m
            for i, m in enumerate(san_history)
        )
        print(f"\n  Moves: {moves_str}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/trainer.pt")
    p.add_argument("--black", action="store_true", help="Play as Black")
    p.add_argument("--sims",  type=int, default=50,
                   help="MCTS simulations per move (default 50, ~10s/move on CPU)")
    args = p.parse_args()

    color  = chess.BLACK if args.black else chess.WHITE
    device = torch.device("cpu")
    play(Path(args.model), color, args.sims, device)
