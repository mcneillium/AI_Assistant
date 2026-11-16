from assistant import model as m
from assistant.data import BOT, USER


def test_prompt_ends_on_the_assistant_turn():
    p = m.build_prompt("how are you?", [("hi", "hello")])
    assert p == f"{USER} hi\n{BOT} hello\n{USER} how are you?\n{BOT}"


def test_reply_stops_at_the_next_turn():
    assert m.clean_reply(" fine thanks!\nUser: and you?") == "fine thanks!"
    assert m.clean_reply(" sure\n\nmore text") == "sure"
    assert m.clean_reply(" i work in a cafe User, what is your job?") == "i work in a cafe"
    assert m.clean_reply(" i relax at the gym. Agent: i play") == "i relax at the gym."
    assert m.clean_reply(" Assistant. hello") == ""


def test_sampling_is_really_on(tiny, monkeypatch):
    """The 2024 bug: sampling settings were passed without do_sample=True."""
    model, tok = tiny
    seen = {}
    real = model.generate

    def spy(**kw):
        seen.update(kw)
        return real(**kw)
    monkeypatch.setattr(model, "generate", spy)
    m.reply(model, tok, "hello", settings=m.GenerationSettings(seed=1, max_new_tokens=5))
    assert seen["do_sample"] is True and seen["top_p"] == 0.9 and seen["max_new_tokens"] == 5


def test_same_seed_same_reply(tiny):
    model, tok = tiny
    s = m.GenerationSettings(seed=3, max_new_tokens=8)
    assert m.reply(model, tok, "hello", settings=s) == m.reply(model, tok, "hello", settings=s)
