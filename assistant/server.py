"""Flask API and web page. Run: python -m assistant.server --model models/gpt2-convai2"""
import argparse
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from . import model as m

MAX_INPUT_CHARS = 500


def create_app(model_dir=None, loaded=None):
    root = Path(__file__).resolve().parent.parent
    app = Flask(__name__, template_folder=str(root / "templates"))
    if loaded is None:
        loaded = m.load(model_dir or os.environ.get("ASSISTANT_MODEL", "models/gpt2-convai2"))
    model, tok = loaded

    @app.after_request
    def cors(resp):  # the React and Streamlit clients run on other local ports
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", model=getattr(model.config, "_name_or_path", ""))

    @app.route("/api/generate", methods=["POST", "OPTIONS"])
    def generate():
        if request.method == "OPTIONS":
            return ("", 204)
        data = request.get_json(silent=True) or {}
        text = str(data.get("input_text", "")).strip()
        if not text:
            return jsonify(error="No input text provided"), 400
        if len(text) > MAX_INPUT_CHARS:
            return jsonify(error=f"Input is longer than {MAX_INPUT_CHARS} characters"), 400
        history = [(str(u), str(a)) for u, a in data.get("history", [])][-4:]
        settings = m.GenerationSettings(seed=data.get("seed"))
        return jsonify(generated_text=m.reply(model, tok, text, history, settings))

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("ASSISTANT_MODEL", "models/gpt2-convai2"))
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    create_app(a.model).run(port=a.port)
