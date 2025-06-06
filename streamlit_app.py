import streamlit as st
from streamlit_chess_component import chessboard

st.title("Chess Bot UI")

# Initialize session FEN
if "fen_from_js" not in st.session_state:
    st.session_state.fen_from_js = "start"

# Show the board, get updated FEN after user move
new_fen = chessboard(fen=st.session_state.fen_from_js)

if new_fen != st.session_state.fen_from_js:
    st.session_state.fen_from_js = new_fen
    st.success(f"User played. New FEN: {new_fen}")
