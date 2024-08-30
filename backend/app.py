from flask import Flask, request, jsonify
from flask_cors import CORS

# Import your AI model or functions here
# from my_ai_model import get_ai_response

app = Flask(__name__)
CORS(app)  # This will enable CORS to allow communication with the React frontend

# Mock function for AI response (replace this with your actual AI function)
def get_ai_response(user_input):
    return f"AI says: {user_input[::-1]}"  # Example: Reverses the input text

@app.route('/api/get_response', methods=['POST'])
def get_response():
    data = request.json
    user_input = data['input']
    response = get_ai_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
