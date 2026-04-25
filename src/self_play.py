"""Batched self-play: generates (state, π, z) training tuples via parallel MCTS-guided games."""

import numpy as np
import torch
import chess
from pathlib import Path

from game import ChessGame, POLICY_SIZE
from mcts import MCTSNode, _run_selection, _do_backup

SelfPlayExample = tuple[np.ndarray, np.ndarray, float]

_TEMP_THRESHOLD = 30   # half-moves before switching to greedy


# ---------------------------------------------------------------------------
# Batched self-play
# ---------------------------------------------------------------------------

def generate_games_batched(
    model: torch.nn.Module,
    device: torch.device,
    num_games: int,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    book_path: Path | None = None,
    max_moves: int = 512,
) -> list[SelfPlayExample]:
    """Run ``num_games`` self-play games in parallel with batched NN inference.

    All games advance one MCTS simulation per outer loop iteration.  At each
    step the leaf positions from every active game are stacked into a single
    tensor and evaluated in **one forward pass** (batch = num_active_games).

    vs. sequential:  250 k individual batch-1 calls  →  ~10 k batched calls
    GPU utilisation: A100 goes from <5 % to >60 % (batch ≈ 100 with default
    num_games=100; scales further up to ~200 before Python becomes the floor).

    bfloat16 autocast is used on CUDA — A100's native precision, 2× throughput
    over fp32 at no accuracy cost for inference.
    """
    # ── per-game state ────────────────────────────────────────────────────
    games    = [ChessGame(book_path=book_path) for _ in range(num_games)]
    roots    = [MCTSNode()                     for _ in range(num_games)]
    sim_cnts = [0] * num_games
    plies    = [0] * num_games
    # (encoded_state, policy, turn) collected while playing; value assigned at end
    bufs: list[list[tuple[np.ndarray, np.ndarray, chess.Color]]] = \
        [[] for _ in range(num_games)]

    active: set[int] = set(range(num_games))   # indices of games still running
    _loop_iter = 0

    # ── main loop: one simulation step per active game per iteration ──────
    while active:
        _loop_iter += 1
        if _loop_iter <= 3 or _loop_iter % 500 == 0:
            print(f"  [dbg] loop iter {_loop_iter}, active={len(active)}", flush=True)

        # ── 1. advance book moves; flush games that have enough sims ─────
        finished: list[int] = []
        for i in list(active):
            g = games[i]

            # consume any book moves without MCTS
            while not g.is_game_over() and plies[i] < max_moves:
                bm = g.book_move()
                if bm is None:
                    break
                g.push(bm)
                plies[i] += 1

            if g.is_game_over() or plies[i] >= max_moves:
                finished.append(i)
                continue

            # enough sims → extract policy, push move, reset tree
            if sim_cnts[i] >= num_simulations:
                root  = roots[i]
                legal = g.legal_move_indices()
                cnts  = np.array(
                    [root.children[idx].visit_count for idx in legal],
                    dtype=np.float32,
                )
                temp = 1.0 if plies[i] < _TEMP_THRESHOLD else 0.0
                if temp < 1e-8:
                    pol = np.zeros(POLICY_SIZE, dtype=np.float32)
                    pol[legal[int(np.argmax(cnts))]] = 1.0
                else:
                    cnts  = cnts ** (1.0 / temp)
                    cnts /= cnts.sum()
                    pol = np.zeros(POLICY_SIZE, dtype=np.float32)
                    for idx, p in zip(legal, cnts):
                        pol[idx] = float(p)

                bufs[i].append((g.encode(), pol, g.turn))

                probs = np.array([pol[idx] for idx in legal], dtype=np.float64)
                probs /= probs.sum() + 1e-12
                g.push_index(int(np.random.choice(legal, p=probs)))
                plies[i] += 1

                roots[i]    = MCTSNode()
                sim_cnts[i] = 0

                if g.is_game_over() or plies[i] >= max_moves:
                    finished.append(i)

        active.difference_update(finished)
        if not active:
            break

        # ── 2. selection: traverse each tree to a leaf (push/pop, no clone) ──
        pending: list[tuple[int, list, list]] = []   # (game_idx, path, actions)

        for i in active:
            path, actions, is_terminal, term_val = _run_selection(
                roots[i], games[i], c_puct
            )
            if is_terminal:
                for _ in actions:
                    games[i].pop()
                _do_backup(path, term_val)
                sim_cnts[i] += 1
            else:
                pending.append((i, path, actions))

        if not pending:
            continue

        # ── 3. batch NN inference (bfloat16 on CUDA) ─────────────────────
        # games[i] is currently AT its leaf state (actions are still pushed)
        states_np = np.stack([games[i].encode() for i, _, _ in pending])
        x = torch.from_numpy(states_np).to(device)

        use_amp = device.type == "cuda"
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits_batch, value_batch = model(x)

        # ── 4. expand nodes, pop actions, backup ─────────────────────────
        # One batched GPU→CPU transfer instead of N individual copies.
        logits_all = logits_batch.float().cpu().numpy()
        values_all = value_batch.squeeze(-1).float().cpu().numpy()

        for k, (i, path, actions) in enumerate(pending):
            g    = games[i]
            node = path[-1]
            is_root_expansion = (node is roots[i])

            # softmax over legal moves (fp32 for numerical safety)
            legal = g.legal_move_indices()
            mask  = np.zeros(POLICY_SIZE, dtype=bool)
            for idx in legal:
                mask[idx] = True
            logits = logits_all[k]
            logits = np.where(mask, logits, -1e9)
            logits -= logits.max()
            exp_l  = np.exp(logits)
            policy = np.where(mask, exp_l, 0.0)
            policy /= policy.sum() + 1e-8

            # Dirichlet noise only at each game's root
            if is_root_expansion and legal:
                noise = np.random.dirichlet([dirichlet_alpha] * len(legal))
                for j, idx in enumerate(legal):
                    policy[idx] = (
                        (1 - dirichlet_epsilon) * policy[idx]
                        + dirichlet_epsilon * noise[j]
                    )
                total = sum(policy[idx] for idx in legal)
                if total > 0:
                    for idx in legal:
                        policy[idx] /= total

            # expand (reuse legal list computed above)
            for idx in legal:
                node.children[idx] = MCTSNode(prior=float(policy[idx]))
            node.is_expanded = True

            if is_root_expansion:
                node.visit_count = 1   # prime (matches MCTS.get_policy)

            value = float(values_all[k])

            # restore game to its root position before backup
            for _ in actions:
                g.pop()

            _do_backup(path, value)
            sim_cnts[i] += 1

    # ── assemble training examples ────────────────────────────────────────
    examples: list[SelfPlayExample] = []
    for i in range(num_games):
        result = games[i].result() or 0.0
        for state, policy, turn in bufs[i]:
            z = float(result) if turn == chess.WHITE else -float(result)
            examples.append((state, policy, z))
    return examples
