"""Entry point kept for anyone used to `python app.py`. The code lives in assistant/."""
from assistant.server import create_app

if __name__ == "__main__":
    create_app().run(port=5000)
