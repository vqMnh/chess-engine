"""Monte Carlo Tree Search with PUCT node selection."""

import math
import numpy as np
import torch

from game import ChessGame, POLICY_SIZE


class MCTSNode:
    """One node in the MCTS tree, corresponding to a board position."""

    __slots__ = ("prior", "visit_count", "value_sum", "children", "is_expanded")

    def __init__(self, prior: float = 1.0):
        self.prior = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: dict[int, "MCTSNode"] = {}
        self.is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def puct_score(self, parent_visits: int, c_puct: float) -> float:
        """PUCT score as seen by the parent (Q is negated; parent is opponent)."""
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return -self.q_value + u


# ---------------------------------------------------------------------------
# Module-level primitives — used by both MCTS and BatchedSelfPlay
# ---------------------------------------------------------------------------

def _puct_select(node: MCTSNode, c_puct: float) -> tuple[int, MCTSNode]:
    """Return the (action, child) with the highest PUCT score."""
    best_score = -math.inf
    best_action = -1
    best_child: MCTSNode | None = None
    for action, child in node.children.items():
        score = child.puct_score(node.visit_count, c_puct)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child  # type: ignore[return-value]


def _run_selection(
    root: MCTSNode,
    game: ChessGame,
    c_puct: float,
) -> tuple[list[MCTSNode], list[int], bool, float]:
    """Traverse from root using PUCT, pushing moves onto *game* in place.

    No clone is made — the caller owns the push/pop contract:
      - On return, game is at the leaf position (actions have been pushed).
      - Caller must call ``game.pop()`` for every element of ``actions``.

    Returns:
        path:           Nodes from root to leaf (inclusive).
        actions:        Move indices pushed onto game (undo with game.pop()).
        is_terminal:    True when the leaf is a game-over state.
        terminal_value: Value at a terminal leaf (current-player perspective).
    """
    node = root
    path = [node]
    actions: list[int] = []

    while node.is_expanded and not game.is_game_over():
        action, node = _puct_select(node, c_puct)
        game.push_index(action)
        actions.append(action)
        path.append(node)

    if game.is_game_over():
        result = game.result()
        return path, actions, True, 0.0 if not result else -1.0
    return path, actions, False, 0.0


def _do_backup(path: list[MCTSNode], value: float) -> None:
    """Propagate value up the path, flipping sign at each edge."""
    for n in reversed(path):
        n.visit_count += 1
        n.value_sum += value
        value = -value


# ---------------------------------------------------------------------------
# Single-game MCTS (used by Evaluator; also kept for single-game self-play)
# ---------------------------------------------------------------------------

class MCTS:
    """AlphaZero-style MCTS guided by a neural network.

    Builds a fresh search tree for each call to get_policy.
    Callers are responsible for putting the model in eval mode before use.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_simulations: int = 800,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    @torch.no_grad()
    def _evaluate(self, game: ChessGame) -> tuple[np.ndarray, float]:
        """Neural-net inference (batch=1). Returns (masked policy, value)."""
        x = torch.from_numpy(game.encode()).unsqueeze(0).to(self.device)
        logits, value = self.model(x)

        logits_np = logits[0].cpu().numpy()
        mask = game.legal_moves_mask()
        logits_np = np.where(mask, logits_np, -1e9)
        logits_np -= logits_np.max()
        exp_l = np.exp(logits_np)
        policy = np.where(mask, exp_l, 0.0)
        policy /= policy.sum() + 1e-8

        return policy, float(value.item())

    def _expand(self, node: MCTSNode, game: ChessGame, add_noise: bool = False) -> float:
        """Evaluate position, populate children with priors, return value estimate."""
        policy, value = self._evaluate(game)
        legal = game.legal_move_indices()

        if add_noise and legal:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
            for i, idx in enumerate(legal):
                policy[idx] = (1 - self.dirichlet_epsilon) * policy[idx] + self.dirichlet_epsilon * noise[i]
            total = sum(policy[idx] for idx in legal)
            if total > 0:
                for idx in legal:
                    policy[idx] /= total

        for idx in legal:
            node.children[idx] = MCTSNode(prior=float(policy[idx]))

        node.is_expanded = True
        return value

    def _simulate(self, root: MCTSNode, game: ChessGame) -> None:
        """One simulation using push/pop — no game.clone() needed."""
        path, actions, is_terminal, terminal_value = _run_selection(root, game, self.c_puct)

        if is_terminal:
            value = terminal_value
        else:
            value = self._expand(path[-1], game)   # game is at leaf state

        for _ in actions:
            game.pop()                             # restore game to root position

        _do_backup(path, value)

    def get_policy(
        self,
        game: ChessGame,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Run MCTS and return an improved policy for the current position.

        Args:
            game:        Position to search. Not mutated.
            temperature: 1.0 = proportional to visit counts; 0.0 = greedy argmax.
            add_noise:   Dirichlet noise at root (True for self-play, False for eval).

        Returns:
            Float32 (POLICY_SIZE,) array, non-zero only at legal moves.
        """
        root = MCTSNode()
        self._expand(root, game, add_noise=add_noise)
        root.visit_count = 1  # prime so first-simulation PUCT exploration is non-zero

        for _ in range(self.num_simulations):
            self._simulate(root, game)

        legal = game.legal_move_indices()
        counts = np.array([root.children[i].visit_count for i in legal], dtype=np.float32)

        if temperature < 1e-8:
            policy = np.zeros(POLICY_SIZE, dtype=np.float32)
            policy[legal[int(np.argmax(counts))]] = 1.0
            return policy

        counts = counts ** (1.0 / temperature)
        counts /= counts.sum()

        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        for idx, prob in zip(legal, counts):
            policy[idx] = float(prob)
        return policy
