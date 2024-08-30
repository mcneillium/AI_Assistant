from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

app = Flask(__name__)

# Load the model and tokenizer
model_dir = './fine-tuned-gpt2'
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_text():
    data = request.json
    input_text = data.get('input_text', '')

    if not input_text:
        return jsonify({'error': 'No input text provided'}), 400

    # Tokenize the input text
    inputs = tokenizer.encode(input_text, return_tensors='pt')

    # Replace the following block with the updated generation code
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_length=50,  # shorter max length
            temperature=0.7,  # moderate temperature for balanced randomness
            top_k=50,  # consider only the top-k logits
            top_p=0.9,  # or use nucleus sampling
            no_repeat_ngram_size=2  # prevent repeating n-grams
        )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return the generated text as a JSON response
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
