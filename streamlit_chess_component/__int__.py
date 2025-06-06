import streamlit as st
import streamlit.components.v1 as components
import os

# Absolute path to the frontend HTML file
frontend_path = os.path.join(os.path.dirname(__file__), "frontend.html")

def chessboard(fen="start", key="chess"):
    with open(frontend_path, "r") as f:
        html = f.read()
    html = html.replace("{{FEN}}", fen)
    result = components.html(html, height=480, key=key)
    return st.session_state.get("fen_from_js", fen)
