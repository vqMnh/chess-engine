"""Chess environment: wraps python-chess to provide board state encoding and move handling."""

import chess
import chess.polyglot
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 6 piece types × 2 players + 4 castling planes + 1 en-passant + 1 turn flag
NUM_PLANES = 18

# 64×64 from/to square pairs + 9 underpromotion slots per from-square (64×9)
# Matches the AlphaZero 4672-action space.
POLICY_SIZE = 4672

_PIECE_PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}
# Underpromotion piece order: Knight=0, Bishop=1, Rook=2
_UNDERPROMO = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


# ---------------------------------------------------------------------------
# Move encoding
# ---------------------------------------------------------------------------

def move_to_index(move: chess.Move, turn: chess.Color = chess.WHITE) -> int:
    """Encode a move as a policy index in [0, POLICY_SIZE).

    Board state and moves are represented from the current player's perspective:
    squares are mirrored vertically when it is Black's turn so the network sees
    a consistent view regardless of color.

    Layout:
      [0, 4096)  — from_sq * 64 + to_sq  (queen promotions fall here)
      [4096, 4672) — underpromotions: from_sq * 9 + col_offset * 3 + piece_off
        col_offset = to_file − from_file + 1  ∈ {0, 1, 2}
        piece_off  = index into [KNIGHT, BISHOP, ROOK]
    """
    from_sq = move.from_square
    to_sq = move.to_square
    if turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    if move.promotion and move.promotion != chess.QUEEN:
        col_offset = chess.square_file(to_sq) - chess.square_file(from_sq) + 1
        piece_off = _UNDERPROMO.index(move.promotion)
        return 4096 + from_sq * 9 + col_offset * 3 + piece_off

    return from_sq * 64 + to_sq


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Decode a policy index back to a chess.Move for the given board position."""
    turn = board.turn
    flip = (turn == chess.BLACK)

    if idx < 4096:
        from_sq_r = idx // 64
        to_sq_r = idx % 64
        from_sq = chess.square_mirror(from_sq_r) if flip else from_sq_r
        to_sq = chess.square_mirror(to_sq_r) if flip else to_sq_r
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_sq)
            if (turn == chess.WHITE and to_rank == 7) or (turn == chess.BLACK and to_rank == 0):
                return chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        return chess.Move(from_sq, to_sq)

    # Underpromotion
    rel = idx - 4096
    from_sq_r = rel // 9
    rem = rel % 9
    col_offset = rem // 3
    piece_off = rem % 3
    from_file = chess.square_file(from_sq_r)
    to_file = from_file + col_offset - 1
    # In relative coords the promotion always lands on rank 7 (current player's back rank)
    to_sq_r = chess.square(to_file, 7)
    from_sq = chess.square_mirror(from_sq_r) if flip else from_sq_r
    to_sq = chess.square_mirror(to_sq_r) if flip else to_sq_r
    return chess.Move(from_sq, to_sq, promotion=_UNDERPROMO[piece_off])


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

def encode_board(board: chess.Board) -> np.ndarray:
    """Return a float32 array of shape (NUM_PLANES, 8, 8) encoding the position.

    Always from the current player's perspective: ranks are flipped when Black
    is to move so the network sees the same orientation regardless of color.

    Plane layout:
      0-5   current player's pieces  (P, N, B, R, Q, K)
      6-11  opponent's pieces        (P, N, B, R, Q, K)
      12    current player kingside castling right
      13    current player queenside castling right
      14    opponent kingside castling right
      15    opponent queenside castling right
      16    en-passant square (single bit)
      17    turn flag (0 = White to move, 1 = Black to move)
    """
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    flip = (board.turn == chess.BLACK)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        if flip:
            rank = 7 - rank
        plane = _PIECE_PLANE[piece.piece_type] + (0 if piece.color == board.turn else 6)
        planes[plane, rank, file] = 1.0

    us, them = board.turn, not board.turn
    planes[12, :, :] = float(board.has_kingside_castling_rights(us))
    planes[13, :, :] = float(board.has_queenside_castling_rights(us))
    planes[14, :, :] = float(board.has_kingside_castling_rights(them))
    planes[15, :, :] = float(board.has_queenside_castling_rights(them))

    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        if flip:
            ep_rank = 7 - ep_rank
        planes[16, ep_rank, ep_file] = 1.0

    planes[17, :, :] = float(board.turn == chess.BLACK)
    return planes


# ---------------------------------------------------------------------------
# Game wrapper
# ---------------------------------------------------------------------------

class ChessGame:
    """Stateful chess game wrapper for the self-play pipeline.

    Provides board encoding, move index conversion, legal-move masks,
    game-over detection, and optional opening-book lookup.
    """

    def __init__(self, book_path: Path | None = None):
        self.board = chess.Board()
        self._book_path = Path(book_path) if book_path is not None else None

    # --- Lifecycle ---

    def reset(self) -> None:
        self.board.reset()

    def __repr__(self) -> str:
        return f"ChessGame(fen={self.board.fen()!r})"

    # --- Board info ---

    @property
    def turn(self) -> chess.Color:
        return self.board.turn

    # --- Move handling ---

    def push(self, move: chess.Move) -> None:
        self.board.push(move)

    def pop(self) -> chess.Move:
        return self.board.pop()

    def push_index(self, idx: int) -> None:
        self.board.push(index_to_move(idx, self.board))

    def legal_move_indices(self) -> list[int]:
        turn = self.board.turn
        return [move_to_index(m, turn) for m in self.board.legal_moves]

    def legal_moves_mask(self) -> np.ndarray:
        """Boolean mask of shape (POLICY_SIZE,) — True at every legal move index."""
        mask = np.zeros(POLICY_SIZE, dtype=bool)
        for idx in self.legal_move_indices():
            mask[idx] = True
        return mask

    # --- Encoding ---

    def encode(self) -> np.ndarray:
        """Board encoding from the current player's perspective, shape (NUM_PLANES, 8, 8)."""
        return encode_board(self.board)

    # --- Termination / result ---

    def is_game_over(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    def result(self) -> float | None:
        """Result from White's perspective: 1=White wins, −1=Black wins, 0=draw.

        Returns None while the game is still in progress.
        Self-play code should flip the sign for Black's training targets.
        """
        if not self.is_game_over():
            return None
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == chess.WHITE else -1.0

    # --- Opening book ---

    def book_move(self) -> chess.Move | None:
        """Return a weighted-random move from the opening book, or None.

        Falls back to None silently if the book file is missing or the position
        has no book entry, so callers can always hand off to MCTS transparently.
        """
        if self._book_path is None or not self._book_path.exists():
            return None
        try:
            with chess.polyglot.open_reader(self._book_path) as reader:
                return reader.weighted_choice(self.board).move
        except IndexError:
            return None
