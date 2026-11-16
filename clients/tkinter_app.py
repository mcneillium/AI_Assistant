"""Tkinter desktop client. Start the API first: python -m assistant.server"""
import tkinter as tk
from tkinter import scrolledtext

import requests

API = "http://127.0.0.1:5000/api/generate"
history = []


def show(text):
    out.config(state=tk.NORMAL)
    out.insert(tk.END, text + "\n\n")
    out.see(tk.END)
    out.config(state=tk.DISABLED)


def send(_event=None):
    text = entry.get().strip()
    if not text:
        return
    entry.delete(0, tk.END)
    show(f"You: {text}")
    try:
        r = requests.post(API, json={"input_text": text, "history": history}, timeout=60)
        data = r.json()
        if r.ok:
            show(f"Assistant: {data['generated_text']}")
            history.append((text, data["generated_text"]))
        else:
            show(f"Error: {data.get('error', r.status_code)}")
    except requests.RequestException as e:
        show(f"Could not reach the API at {API}: {e}")


root = tk.Tk()
root.title("AI Assistant")
out = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, state=tk.DISABLED)
out.pack(padx=10, pady=10)
entry = tk.Entry(root, width=60)
entry.pack(padx=10, pady=(0, 10))
entry.bind("<Return>", send)
tk.Button(root, text="Send", command=send).pack(pady=(0, 10))
entry.focus()
root.mainloop()
