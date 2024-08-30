import tkinter as tk
from tkinter import scrolledtext
import requests

def generate_text():
    input_text = input_box.get("1.0", tk.END)
    response = requests.post("http://127.0.0.1:5000/api/generate", json={"input_text": input_text.strip()})
    result = response.json()
    
    # Updated code block
    if 'generated_text' in result:
        output_box.config(state=tk.NORMAL)
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, result['generated_text'])
        output_box.config(state=tk.DISABLED)
    else:
        output_box.config(state=tk.NORMAL)
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, "Error occurred: " + result.get('error', 'Unknown error'))
        output_box.config(state=tk.DISABLED)

root = tk.Tk()
root.title("AI Assistant")

input_label = tk.Label(root, text="Enter your prompt:")
input_label.pack()
input_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10)
input_box.pack()

generate_button = tk.Button(root, text="Generate", command=generate_text)
generate_button.pack()

output_label = tk.Label(root, text="Generated Text:")
output_label.pack()
output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10, state=tk.DISABLED)
output_box.pack()

root.mainloop()