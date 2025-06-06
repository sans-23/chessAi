import streamlit as st
import chess
import chess.engine
import torch
import numpy as np
from fen2eval.preprocessing import fen_to_input, denormalize_cp
from fen2eval.model import FEN2EvalCNN
from streamlit_chessboard import chessboard  # pip install streamlit-chessboard

# --- Load your AI model ---
@st.cache_resource
def load_model():
    model = FEN2EvalCNN()
    model.load_state_dict(torch.load("model_weights_32.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# --- Initialize Stockfish engine ---
@st.cache_resource
def init_stockfish():
    # Replace 'stockfish' with your actual path if needed
    return chess.engine.SimpleEngine.popen_uci("stockfish")

engine = init_stockfish()

st.title("FEN2Eval Chess Bot")

# User chooses color
user_color = st.radio("Choose your side:", ("White", "Black"))
user_plays_white = user_color == "White"

# Initialize or get current board state from session
if "board_fen" not in st.session_state:
    st.session_state.board_fen = chess.STARTING_FEN
    st.session_state.move_count = 0  # count total half-moves

board = chess.Board(st.session_state.board_fen)

# Display interactive chessboard, user can only move their pieces
# The chessboard component returns the new FEN after user move
new_fen = chessboard(
    fen=st.session_state.board_fen,
    size=400,
    orientation="white" if user_plays_white else "black",
    # only allow user to move their own color
    player_color="white" if user_plays_white else "black",
    theme="blue",  # optional styling
)

# Detect user move (FEN changed and move is legal)
if new_fen != st.session_state.board_fen:
    try:
        new_board = chess.Board(new_fen)
        # Check if the move was legal and user moved their piece
        if new_board.move_stack and board.turn == (user_plays_white):
            last_move = new_board.peek()
            if board.is_legal(last_move):
                # Accept user move
                board.push(last_move)
                st.session_state.move_count += 1
            else:
                st.warning("Illegal move! Please try again.")
        else:
            st.warning("It's not your turn!")
    except Exception as e:
        st.error(f"Error processing your move: {e}")

# After user move, let AI or Stockfish play moves until it's user's turn again or game ends
def ai_move(board):
    """AI bot move: evaluate all legal moves, pick best eval for the bot side."""
    legal_moves = list(board.legal_moves)
    best_move = None
    best_eval = -float("inf") if board.turn == chess.WHITE else float("inf")
    bot_color = board.turn

    for move in legal_moves:
        board.push(move)
        fen = board.fen()
        try:
            board_tensor, extras = fen_to_input(fen)
            board_tensor = torch.tensor(board_tensor).unsqueeze(0).float()
            extras = torch.tensor(extras).unsqueeze(0).float()
            with torch.no_grad():
                eval_norm = model(board_tensor, extras).item()
                cp_eval = denormalize_cp(eval_norm)
                # From bot's perspective, maximize winning chances
                # So White tries to maximize cp_eval, Black tries to minimize
                score = cp_eval if bot_color == chess.WHITE else -cp_eval
                if score > best_eval:
                    best_eval = score
                    best_move = move
        except Exception as e:
            st.error(f"Error evaluating move {move}: {e}")
        board.pop()

    return best_move

def stockfish_move(board):
    """Use Stockfish to get best move."""
    result = engine.play(board, chess.engine.Limit(time=0.1))
    return result.move

# Play moves for AI and Stockfish while it's their turn and game not over
while not board.is_game_over():
    turn_is_user = (board.turn == chess.WHITE and user_plays_white) or (board.turn == chess.BLACK and not user_plays_white)
    if turn_is_user:
        break  # Wait for user move

    # Determine which bot to play: odd moves by AI model, even by Stockfish
    # move_count counts half moves made so far
    # After user move, move_count incremented so odd means AI bot
    if st.session_state.move_count % 2 == 1:
        # AI move
        move = ai_move(board)
        if move is None:
            st.warning("AI bot could not find a move!")
            break
        board.push(move)
        st.session_state.move_count += 1
        st.info(f"AI bot moves: {board.san(move)}")
    else:
        # Stockfish move
        move = stockfish_move(board)
        if move is None:
            st.warning("Stockfish could not find a move!")
            break
        board.push(move)
        st.session_state.move_count += 1
        st.info(f"Stockfish moves: {board.san(move)}")

# Update fen in session
st.session_state.board_fen = board.fen()

# Show final board after all moves played
st.subheader("Current Position")
chessboard(fen=board.fen(), size=400, orientation="white" if user_plays_white else "black", player_color=None)

# Show game status
if board.is_checkmate():
    st.error(f"Checkmate! {'White' if board.turn == chess.BLACK else 'Black'} wins.")
elif board.is_stalemate():
    st.warning("Game drawn by stalemate.")
elif board.is_insufficient_material():
    st.warning("Game drawn by insufficient material.")
elif board.is_seventyfive_moves():
    st.warning("Game drawn by 75-move rule.")
elif board.is_fivefold_repetition():
    st.warning("Game drawn by fivefold repetition.")
else:
    # Show evaluation of current position from bot perspective
    try:
        board_tensor, extras = fen_to_input(board.fen())
        board_tensor = torch.tensor(board_tensor).unsqueeze(0).float()
        extras = torch.tensor(extras).unsqueeze(0).float()
        with torch.no_grad():
            eval_norm = model(board_tensor, extras).item()
            cp_eval = denormalize_cp(eval_norm)
        st.write(f"Current evaluation (cp): {cp_eval:.1f} (positive = White advantage)")
    except Exception as e:
        st.error(f"Error evaluating current position: {e}")

# Close Stockfish engine on exit (optional)
# You can also handle this in a better lifecycle way if needed
# engine.quit()
