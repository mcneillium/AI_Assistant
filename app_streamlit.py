import streamlit as st
import requests

st.title("AI Assistant")

input_text = st.text_area("Enter your prompt:")

if st.button("Generate"):
    if input_text:
        response = requests.post("http://127.0.0.1:5000/api/generate", json={"input_text": input_text})
        result = response.json()
        st.write(result.get("generated_text", "Error occurred"))
    else:
        st.write("Please enter some text.")
