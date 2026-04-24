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
        """Neural-net inference on the current position.

        Returns (policy, value) where policy is a masked, normalised probability
        array of shape (POLICY_SIZE,) and value is a scalar in (−1, 1) from the
        perspective of the player to move.
        """
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

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        """Return the (action, child) with the highest PUCT score."""
        best_score = -math.inf
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            score = child.puct_score(node.visit_count, self.c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child  # type: ignore[return-value]

    def _simulate(self, root: MCTSNode, game: ChessGame) -> None:
        """Run one simulation: select → expand → backup."""
        node = root
        sim_game = game.clone()
        path = [node]

        # Selection: follow PUCT until an unexpanded or terminal node
        while node.is_expanded and not sim_game.is_game_over():
            action, node = self._select_child(node)
            sim_game.push_index(action)
            path.append(node)

        # Evaluation
        if sim_game.is_game_over():
            result = sim_game.result()
            # None or 0.0 → draw; any non-zero result → current player lost
            # (at game-over the player to move is checkmated or stalemated)
            value = 0.0 if not result else -1.0
        else:
            value = self._expand(node, sim_game)

        # Backup: propagate value upward, flipping sign at each edge.
        # node.value_sum accumulates values from that node's player's perspective,
        # so the parent sees the negation.
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    def get_policy(
        self,
        game: ChessGame,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Run MCTS from the current position and return an improved policy.

        Args:
            game:        Position to search. Not mutated.
            temperature: 1.0 = proportional to visit counts (early-game exploration);
                         0.0 = greedy argmax (evaluation / late game).
            add_noise:   Inject Dirichlet noise at root. Use True during self-play,
                         False during head-to-head evaluation.

        Returns:
            Float32 array of shape (POLICY_SIZE,), non-zero only at legal moves.
        """
        root = MCTSNode()
        self._expand(root, game, add_noise=add_noise)
        # Prime visit count so the first simulation's PUCT exploration is non-zero.
        root.visit_count = 1

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
