"""Streamlit client. Start the API first: python -m assistant.server"""
import requests
import streamlit as st

API = "http://127.0.0.1:5000/api/generate"

st.title("AI Assistant")
if "history" not in st.session_state:
    st.session_state.history = []
for u, a in st.session_state.history:
    st.chat_message("user").write(u)
    st.chat_message("assistant").write(a)
if text := st.chat_input("Say something"):
    st.chat_message("user").write(text)
    try:
        r = requests.post(API, json={"input_text": text, "history": st.session_state.history}, timeout=60)
        r.raise_for_status()
        answer = r.json()["generated_text"]
    except requests.RequestException as e:
        st.error(f"Could not reach the API at {API}: {e}")
    else:
        st.chat_message("assistant").write(answer)
        st.session_state.history.append((text, answer))
