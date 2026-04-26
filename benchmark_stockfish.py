#!/usr/bin/env python3
"""
Benchmark the chess engine against Stockfish at multiple ELO levels.

Usage:
    python benchmark_stockfish.py                        # trainer.pt, 20 games, 100 sims
    python benchmark_stockfish.py --model checkpoints/best.pt
    python benchmark_stockfish.py --games 10 --sims 50  # faster run
    python benchmark_stockfish.py --levels 800 1000     # custom ELO targets

Requires: brew install stockfish
"""

import sys
import argparse
from pathlib import Path

import torch
import chess
import chess.engine
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from model import ChessNet
from mcts import MCTS
from game import ChessGame


def load_model(path: Path, device: torch.device) -> ChessNet:
    data = torch.load(path, map_location=device)
    # trainer.pt uses "model_state"; best.pt uses "state_dict"
    state_dict = data.get("model_state") or data.get("state_dict")
    net = ChessNet(
        num_blocks=data["num_blocks"],
        num_filters=data["num_filters"],
    ).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def configure_strength(engine: chess.engine.SimpleEngine, elo: int) -> str:
    """Set Stockfish strength. Returns label describing what was actually set."""
    opts = engine.options
    elo_min = opts["UCI_Elo"].min if "UCI_Elo" in opts else 9999
    if elo >= elo_min:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        return f"{elo} ELO"
    # Below UCI_Elo minimum — fall back to Skill Level (rough mapping)
    skill = max(0, min(20, round((elo - 800) / 100)))
    engine.configure({"UCI_LimitStrength": False, "Skill Level": skill})
    return f"Skill {skill} (~{elo} ELO)"


def play_game(
    mcts: MCTS,
    engine: chess.engine.SimpleEngine,
    elo: int,
    our_color: chess.Color,
    max_moves: int,
    book_path: Path | None,
) -> float:
    """One game vs Stockfish. Returns +1 win / 0 draw / -1 loss from our perspective."""
    game = ChessGame(book_path=book_path)
    configure_strength(engine, elo)

    ply = 0
    while not game.is_game_over() and ply < max_moves:
        bm = game.book_move()
        if bm is not None:
            game.push(bm)
            ply += 1
            continue

        if game.board.turn == our_color:
            policy = mcts.get_policy(game, temperature=0.0, add_noise=False)
            game.push_index(int(np.argmax(policy)))
        else:
            sf_result = engine.play(game.board, chess.engine.Limit(time=0.1))
            game.push(sf_result.move)
        ply += 1

    outcome = game.result()
    score = outcome if outcome is not None else 0.0
    return score if our_color == chess.WHITE else -score


def run_benchmark(
    model_path: Path,
    games_per_level: int,
    sims: int,
    max_moves: int,
    book_path: Path | None,
    elo_levels: list[int],
    device: torch.device,
) -> None:
    print(f"\nLoading model from {model_path} ...")
    net = load_model(model_path, device)
    mcts = MCTS(net, device, num_simulations=sims, c_puct=1.5)

    print(f"Device : {device}")
    print(f"Sims   : {sims}  |  Games/level : {games_per_level}  |  Max moves : {max_moves}")
    print(f"Levels : {elo_levels}\n")

    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    half = games_per_level // 2

    # Show actual minimum ELO so user knows what Skill Level fallback kicks in
    elo_min = engine.options["UCI_Elo"].min if "UCI_Elo" in engine.options else None
    if elo_min:
        print(f"Stockfish UCI_Elo minimum: {elo_min}  (levels below use Skill Level)\n")

    results = {}
    try:
        for elo in elo_levels:
            wins = draws = losses = 0
            game_num = 0
            strength_label = configure_strength(engine, elo)

            for color, clabel in [(chess.WHITE, "W"), (chess.BLACK, "B")]:
                n = half if color == chess.WHITE else games_per_level - half
                for _ in range(n):
                    game_num += 1
                    r = play_game(mcts, engine, elo, color, max_moves, book_path)
                    symbol = "W" if r > 0 else ("D" if r == 0 else "L")
                    if r > 0:    wins   += 1
                    elif r == 0: draws  += 1
                    else:        losses += 1
                    print(
                        f"  vs {strength_label:<18}  game {game_num:>2}/{games_per_level}"
                        f"  as {clabel}  → {symbol}"
                        f"  [{wins}W {draws}D {losses}L]",
                        flush=True,
                    )

            total    = wins + draws + losses
            win_rate = (wins + 0.5 * draws) / total
            results[elo] = (wins, draws, losses, win_rate)
            print()

    finally:
        engine.quit()

    # Summary table
    print("=" * 54)
    print(f"{'ELO':>6}  {'W':>4}  {'D':>4}  {'L':>4}  {'Score':>7}  Verdict")
    print("-" * 54)
    for elo, (w, d, l, wr) in results.items():
        verdict = "BEATS" if wr > 0.55 else ("DRAWS" if wr >= 0.45 else "LOSES TO")
        print(f"{elo:>6}  {w:>4}  {d:>4}  {l:>4}  {wr:>7.3f}  {verdict}")
    print("=" * 54)

    # ELO estimate
    beating = [elo for elo, (_, _, _, wr) in results.items() if wr > 0.55]
    losing  = [elo for elo, (_, _, _, wr) in results.items() if wr < 0.45]
    if beating and losing:
        print(f"\nEstimated ELO: {max(beating)}–{min(losing)}")
    elif beating:
        print(f"\nEstimated ELO: >{max(beating)}")
    elif losing:
        print(f"\nEstimated ELO: <{min(losing)}")
    else:
        print(f"\nEstimated ELO: ~{elo_levels[len(elo_levels)//2]} (all draws — narrow bracket)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="checkpoints/trainer.pt")
    parser.add_argument("--games",     type=int,   default=20)
    parser.add_argument("--sims",      type=int,   default=100)
    parser.add_argument("--max-moves", type=int,   default=200)
    parser.add_argument("--book",      default="books/gm2001.bin")
    parser.add_argument("--levels",    type=int,   nargs="+", default=[800, 1000, 1200, 1400])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book   = Path(args.book) if Path(args.book).exists() else None

    run_benchmark(
        model_path     = Path(args.model),
        games_per_level= args.games,
        sims           = args.sims,
        max_moves      = args.max_moves,
        book_path      = book,
        elo_levels     = args.levels,
        device         = device,
    )
