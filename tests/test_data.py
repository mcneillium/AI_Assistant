from assistant.data import BOT, USER, format_dialogue, load_custom, split


def test_roles_and_whitespace():
    turns = [{"sender_class": "Human", "text": "hi  there\n"}, {"sender_class": "Bot", "text": "hello"},
             {"sender_class": "Human", "text": "   "}]
    assert format_dialogue(turns) == f"{USER} hi there\n{BOT} hello"


def test_split_is_by_dialogue_and_repeatable():
    items = [f"d{i}" for i in range(100)]
    tr1, ev1 = split(items, 0.1, seed=7)
    tr2, ev2 = split(items, 0.1, seed=7)
    assert (tr1, ev1) == (tr2, ev2)
    assert len(ev1) == 10 and not set(tr1) & set(ev1) and len(tr1) + len(ev1) == 100


def test_custom_dialogues(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"turns": [{"role": "user", "text": "Where is the bus?"}, '
                 '{"role": "assistant", "text": "At stop 4."}]}\n')
    assert load_custom(p) == [f"{USER} Where is the bus?\n{BOT} At stop 4."]
