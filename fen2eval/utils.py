import torch
import chess
import numpy as np

def fen_to_tensor(fen: str):
    # your FEN parsing logic to convert into model input tensor
    ...

def get_best_move(fen: str, model):
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    best_score = -float('inf')
    best_move = None

    for move in legal_moves:
        board.push(move)
        tensor_input = fen_to_tensor(board.fen())
        with torch.no_grad():
            score = model(tensor_input).item()
        if score > best_score:
            best_score = score
            best_move = move
        board.pop()

    return best_move.uci(), best_score
