from __future__ import annotations

import json
from pathlib import Path

import chess


_PROMOTION_PIECES_LC0_ORDER = (chess.ROOK, chess.BISHOP, chess.QUEEN)


def _is_queen_or_knight_move(from_square: chess.Square, to_square: chess.Square) -> bool:
    if from_square == to_square:
        return False

    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = chess.square_file(to_square)
    to_rank = chess.square_rank(to_square)

    df = to_file - from_file
    dr = to_rank - from_rank
    adf = abs(df)
    adr = abs(dr)

    # "Queen" part in AlphaZero/Lc0 move-planes means every ray move.
    if df == 0 or dr == 0 or adf == adr:
        return True

    # 8 knight jumps.
    return (adf, adr) in {(1, 2), (2, 1)}


def generate_lc0_policy_index_to_uci() -> list[str]:
    """Generate Lc0 1858 policy mapping in canonical index order.

    The order follows the Lc0 1858 policy vector convention:
    1) 1792 non-promotion "queen+knight geometry" moves from all 64 squares.
    2) 66 underpromotions (rook, bishop, queen) from rank-7 with deltas -1/0/+1.

    Note: Lc0 treats knight promotion as the base move (without suffix), which is why
    only 3 promotion suffixes are added in the final 66 entries.
    """
    mapping: list[str] = []

    squares = list(chess.SQUARES)

    for from_square in squares:
        from_uci = chess.square_name(from_square)
        for to_square in squares:
            if not _is_queen_or_knight_move(from_square, to_square):
                continue
            to_uci = chess.square_name(to_square)
            mapping.append(f"{from_uci}{to_uci}")

    for from_file in range(8):
        from_square = chess.square(from_file, 6)  # 7th rank in white-canonical view
        from_uci = chess.square_name(from_square)
        for file_delta in (-1, 0, 1):
            to_file = from_file + file_delta
            if to_file < 0 or to_file > 7:
                continue
            to_square = chess.square(to_file, 7)  # 8th rank
            to_uci = chess.square_name(to_square)
            for promotion_piece in _PROMOTION_PIECES_LC0_ORDER:
                promotion_char = chess.piece_symbol(promotion_piece)
                mapping.append(f"{from_uci}{to_uci}{promotion_char}")

    if len(mapping) != 1858:
        raise ValueError(f"Invalid Lc0 policy mapping size: expected 1858, got {len(mapping)}")

    # Sanity anchors guard against accidental order drift.
    expected_prefix = ["a1b1", "a1c1", "a1d1", "a1e1", "a1f1"]
    if mapping[:5] != expected_prefix:
        raise ValueError("Lc0 mapping prefix mismatch")
    if mapping[1791] != "h8g8":
        raise ValueError("Lc0 base segment tail mismatch")
    if mapping[1792] != "a7a8r":
        raise ValueError("Lc0 promotion segment start mismatch")
    if mapping[-1] != "h7h8q":
        raise ValueError("Lc0 promotion segment tail mismatch")

    return mapping


def validate_lc0_policy_index_to_uci(mapping: list[str]) -> None:
    """Validate that mapping is exactly the canonical Lc0-compatible 1858 order."""
    if len(mapping) != 1858:
        raise ValueError(f"Invalid mapping length: expected 1858, got {len(mapping)}")

    if len(set(mapping)) != len(mapping):
        raise ValueError("Invalid mapping: duplicate UCI moves found")

    for idx, uci in enumerate(mapping):
        _validate_uci_shape_for_index(idx, uci)
        try:
            chess.Move.from_uci(uci)
        except ValueError as error:
            raise ValueError(f"Invalid UCI move at index {idx}: {uci}") from error

    expected = generate_lc0_policy_index_to_uci()
    if mapping != expected:
        raise ValueError(
            "Invalid Lc0 policy map order: mapping does not match canonical 1858 sequence"
        )


def _validate_uci_shape_for_index(index: int, uci: str) -> None:
    if index < 1792:
        if len(uci) != 4:
            raise ValueError(f"Invalid base move UCI at index {index}: {uci}")
        from_sq = chess.parse_square(uci[:2])
        to_sq = chess.parse_square(uci[2:4])
        if not _is_queen_or_knight_move(from_sq, to_sq):
            raise ValueError(f"Illegal Lc0 base geometry at index {index}: {uci}")
        return

    if len(uci) != 5:
        raise ValueError(f"Invalid promotion move UCI at index {index}: {uci}")
    if uci[4] not in {"r", "b", "q"}:
        raise ValueError(f"Invalid promotion suffix at index {index}: {uci}")

    from_sq = chess.parse_square(uci[:2])
    to_sq = chess.parse_square(uci[2:4])
    from_rank = chess.square_rank(from_sq)
    to_rank = chess.square_rank(to_sq)
    if from_rank != 6 or to_rank != 7:
        raise ValueError(f"Invalid promotion ranks at index {index}: {uci}")

    file_delta = abs(chess.square_file(to_sq) - chess.square_file(from_sq))
    if file_delta > 1:
        raise ValueError(f"Invalid promotion file delta at index {index}: {uci}")


def write_lc0_policy_index_json(output_path: str | Path) -> None:
    mapping = generate_lc0_policy_index_to_uci()
    validate_lc0_policy_index_to_uci(mapping)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
