from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

from .model_loader import OnnxPolicyModel
from .policy_map import validate_lc0_policy_index_to_uci


PIECE_CHANNEL = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}


def _canonical_coords(square: chess.Square, turn: chess.Color) -> tuple[int, int]:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    if turn == chess.BLACK:
        return 7 - file_idx, 7 - rank_idx
    return file_idx, rank_idx


def fen_to_tensor(board: chess.Board) -> np.ndarray:
    """Encode a FEN position to a simple policy-network tensor [1, 18, 8, 8]."""
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        file_idx, rank_idx = _canonical_coords(square, board.turn)
        channel = PIECE_CHANNEL[(piece.color, piece.piece_type)]
        planes[channel, rank_idx, file_idx] = 1.0

    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    planes[17, :, :] = min(board.halfmove_clock / 100.0, 1.0)

    return np.expand_dims(planes, axis=0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    stabilized = logits - np.max(logits)
    exp_values = np.exp(stabilized)
    denom = np.sum(exp_values)
    if denom == 0:
        return np.zeros_like(logits)
    return exp_values / denom


class PolicyMapper:
    """Maps model policy indexes to chess moves.

    - For policy size 4672: uses AlphaZero-like move planes.
    - For policy size 1858: requires JSON mapping from MAIA_POLICY_MAP_PATH.
    """

    def __init__(self, policy_size: int, mapping_path: str | None = None):
        self.policy_size = policy_size
        self._index_to_uci: dict[int, str] = {}
        self._uci_to_index: dict[str, int] = {}
        self.mapping_path = mapping_path
        self.loaded_from_file = False

        if mapping_path:
            self._load_mapping(mapping_path)
            self.loaded_from_file = True
        elif self.policy_size == 1858:
            raise ValueError(
                "Policy size 1858 detected but MAIA_POLICY_MAP_PATH is not set. "
                "Provide a valid Lc0 policy_index_to_uci.json file."
            )

    def _load_mapping(self, mapping_path: str) -> None:
        payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            mapping = payload
        elif isinstance(payload, dict):
            mapping = self._normalize_mapping_dict(payload)
        else:
            raise ValueError("Policy mapping must be a JSON list or dict")

        if len(mapping) != self.policy_size:
            raise ValueError(
                f"Policy mapping length mismatch: expected {self.policy_size}, got {len(mapping)}"
            )

        if self.policy_size == 1858:
            validate_lc0_policy_index_to_uci(mapping)

        self._index_to_uci = {idx: uci for idx, uci in enumerate(mapping)}
        self._uci_to_index = {uci: idx for idx, uci in self._index_to_uci.items()}
        self._validate_bidirectional_consistency()

    @staticmethod
    def _normalize_mapping_dict(payload: dict) -> list[str]:
        try:
            parsed = {int(idx): uci for idx, uci in payload.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("Policy mapping dict keys must be integer-like indexes") from error

        expected_keys = set(range(len(parsed)))
        if set(parsed.keys()) != expected_keys:
            raise ValueError("Policy mapping dict keys must be contiguous indexes from 0")
        return [parsed[idx] for idx in range(len(parsed))]

    def export_loaded_mapping(self) -> list[str]:
        if not self.loaded_from_file:
            raise ValueError("No mapping loaded from MAIA_POLICY_MAP_PATH")
        if not self._index_to_uci:
            raise ValueError("Loaded mapping is empty")
        return [self._index_to_uci[idx] for idx in range(len(self._index_to_uci))]

    def index_for_uci(self, uci: str) -> int | None:
        return self._uci_to_index.get(uci)

    def uci_for_index(self, index: int) -> str | None:
        return self._index_to_uci.get(index)

    def _validate_bidirectional_consistency(self) -> None:
        if len(self._index_to_uci) != self.policy_size:
            raise ValueError(
                f"Invalid loaded mapping size: expected {self.policy_size}, got {len(self._index_to_uci)}"
            )
        if len(self._uci_to_index) != len(self._index_to_uci):
            raise ValueError("Invalid loaded mapping: duplicate UCI moves detected")

        for index, uci in self._index_to_uci.items():
            back_index = self._uci_to_index.get(uci)
            if back_index != index:
                raise ValueError(
                    f"Invalid loaded mapping: inconsistent round-trip for index {index} and move {uci}"
                )

    def move_to_index(self, move: chess.Move, board: chess.Board) -> int | None:
        if self._uci_to_index:
            uci = move.uci()
            idx = self._uci_to_index.get(uci)
            if idx is not None:
                return idx

            # Lc0 policy convention encodes knight promotion as base move (no suffix).
            if move.promotion == chess.KNIGHT:
                return self._uci_to_index.get(uci[:4])

            return None

        if self.policy_size == 4672:
            return self._alphazero_move_to_index(move, board.turn)

        return None

    def legal_move_distribution(
        self, policy_logits: np.ndarray, board: chess.Board, temperature: float = 1.0
    ) -> list[tuple[chess.Move, float]]:
        moves: list[chess.Move] = []
        logits: list[float] = []

        for move in board.legal_moves:
            idx = self.move_to_index(move, board)
            if idx is None or idx < 0 or idx >= policy_logits.size:
                continue
            moves.append(move)
            logits.append(float(policy_logits[idx]))

        if not moves:
            return []

        legal_logits = np.asarray(logits, dtype=np.float32)
        safe_temp = max(temperature, 1e-6)
        probs = _softmax(legal_logits / safe_temp)

        ranked = sorted(zip(moves, probs.tolist()), key=lambda item: item[1], reverse=True)
        return ranked

    @staticmethod
    def _alphazero_move_to_index(move: chess.Move, turn: chess.Color) -> int | None:
        from_file, from_rank = _canonical_coords(move.from_square, turn)
        to_file, to_rank = _canonical_coords(move.to_square, turn)
        df = to_file - from_file
        dr = to_rank - from_rank

        plane = PolicyMapper._move_plane(df, dr, move.promotion)
        if plane is None:
            return None

        from_index = from_rank * 8 + from_file
        return from_index * 73 + plane

    @staticmethod
    def _move_plane(df: int, dr: int, promotion: int | None) -> int | None:
        directions = [
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
        ]

        if promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
            if dr != 1:
                return None
            promo_dirs = {-1: 0, 0: 1, 1: 2}
            promo_pieces = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
            if df not in promo_dirs:
                return None
            return 64 + promo_dirs[df] * 3 + promo_pieces[promotion]

        for direction_idx, (dir_file, dir_rank) in enumerate(directions):
            for step in range(1, 8):
                if (df, dr) == (dir_file * step, dir_rank * step):
                    return direction_idx * 7 + (step - 1)

        knight_moves = [
            (1, 2),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ]
        for knight_idx, knight_delta in enumerate(knight_moves):
            if (df, dr) == knight_delta:
                return 56 + knight_idx

        return None


@dataclass
class MaiaInferenceEngine:
    model: OnnxPolicyModel
    mapper: PolicyMapper

    @classmethod
    def from_onnx(cls, model_path: str, policy_map_path: str | None = None) -> "MaiaInferenceEngine":
        model = OnnxPolicyModel.from_path(model_path)

        # Dry-run to infer policy output size without requiring a real request.
        probe_board = chess.Board()
        probe_tensor = fen_to_tensor(probe_board)
        probe_logits = model.predict(probe_tensor)
        mapper = PolicyMapper(policy_size=probe_logits.size, mapping_path=policy_map_path)
        return cls(model=model, mapper=mapper)

    def analyze(self, fen: str, top_k: int, temperature: float) -> list[dict[str, float | str]]:
        board = chess.Board(fen)
        board_tensor = fen_to_tensor(board)
        policy_logits = self.model.predict(board_tensor)
        ranked_moves = self.mapper.legal_move_distribution(policy_logits, board, temperature=temperature)

        return [
            {"uci": move.uci(), "prob": round(prob, 6)}
            for move, prob in ranked_moves[: max(1, top_k)]
        ]
