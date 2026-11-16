import pytest

from assistant.server import create_app


@pytest.fixture()
def client(tiny):
    return create_app(loaded=tiny).test_client()


def test_health(client):
    assert client.get("/api/health").get_json()["status"] == "ok"


def test_page(client):
    assert b"AI Assistant" in client.get("/").data


def test_empty_and_too_long_inputs_are_refused(client):
    assert client.post("/api/generate", json={"input_text": "  "}).status_code == 400
    assert client.post("/api/generate", json={}).status_code == 400
    assert client.post("/api/generate", json={"input_text": "x" * 501}).status_code == 400


def test_generate_returns_text(client):
    r = client.post("/api/generate", json={"input_text": "hi", "seed": 1, "history": [["a", "b"]]})
    assert r.status_code == 200 and isinstance(r.get_json()["generated_text"], str)


def test_seed_repeats_a_reply_and_bad_seeds_are_ignored(client):
    ask = lambda seed: client.post("/api/generate", json={"input_text": "hello", "seed": seed}).get_json()["generated_text"]
    assert ask(5) == ask(5)
    for bad in ["5", 1.5, True, -1, 2**40]:
        assert client.post("/api/generate", json={"input_text": "hello", "seed": bad}).status_code == 200
