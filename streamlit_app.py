import streamlit as st
import chess
import torch
import numpy as np
from fen2eval.preprocessing import fen_to_input, denormalize_cp
from fen2eval.model import FEN2EvalCNN

st.title("FEN2Eval: Predict Chess Position Evaluation")

# Load model
@st.cache_resource
def load_model():
    model = FEN2EvalCNN()
    model.load_state_dict(torch.load("model_weights_32.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

fen_input = st.text_input(
    "Enter a FEN string (e.g. `r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4`):",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4"
)

if st.button("Evaluate"):
    try:
        board_tensor, extras = fen_to_input(fen_input)
        board_tensor = torch.tensor(board_tensor).unsqueeze(0)
        extras = torch.tensor(extras).unsqueeze(0)
        with torch.no_grad():
            eval_score = model(board_tensor.float(), extras.float()).item()
            cp_score = denormalize_cp(eval_score)
            st.success(f"Predicted evaluation: {cp_score}")
    except Exception as e:
        st.error(f"Invalid FEN or internal error: {e}")
