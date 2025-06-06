# preprocessing.py

import chess
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        idx = PIECE_TO_INDEX[piece.symbol()]
        tensor[idx, row, col] = 1
    return tensor

def extract_extra_features(board):
    extras = []

    # Turn
    extras.append(1.0 if board.turn == chess.WHITE else 0.0)

    # Castling rights
    extras.extend([
        float(board.has_kingside_castling_rights(chess.WHITE)),
        float(board.has_queenside_castling_rights(chess.WHITE)),
        float(board.has_kingside_castling_rights(chess.BLACK)),
        float(board.has_queenside_castling_rights(chess.BLACK)),
    ])

    # En passant
    ep_onehot = [0.0] * 8
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        ep_onehot[file] = 1.0
    extras.extend(ep_onehot)

    # Halfmove clock (normalize to [0, 1])
    extras.append(min(board.halfmove_clock, 100) / 100.0)

    # Fullmove number (normalize to [0, 1])
    extras.append(min(board.fullmove_number, 100) / 100.0)

    return np.array(extras, dtype=np.float32)

def fen_to_input(fen):
    board = chess.Board(fen)
    board_tensor = board_to_tensor(board)  # Shape: (12, 8, 8)F
    extras = extract_extra_features(board) # Shape: (14,)
    return board_tensor, extras

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    X_board = []
    X_extra = []
    y = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Loading dataset"):
        fen = row["FEN"]
        eval_score = row["Evaluation"]
        board_tensor, extras = fen_to_input(fen)
        X_board.append(board_tensor)
        X_extra.append(extras)

        eval_score_str = str(eval_score).strip()
        if eval_score_str.startswith('#'):
            mate_in = int(eval_score_str[1:])
            y.append(1000.0 - mate_in)
        elif eval_score_str.startswith('-#'):
            mate_in = int(eval_score_str[2:])
            y.append(-1000.0 + mate_in)
        else:
            y.append(float(eval_score_str))

        if i==423123:
          break

    return (
        np.stack(X_board),     # (N, 12, 8, 8)
        np.stack(X_extra),     # (N, 14)
        np.array(y, dtype=np.float32)  # (N,)
    )
