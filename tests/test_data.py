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


def test_detok_undoes_dailydialog_spacing():
    from assistant.data import detok
    assert detok("Say , Jim , how ' s it going ?") == "Say, Jim, how's it going?"
    assert detok("I don ' t know . It costs $ 5 ( maybe ) !") == "I don't know. It costs $5 (maybe)!"


def test_dailydialog_and_bst_loaders(tmp_path):
    import pandas as pd
    from assistant.data import load_bst, load_dailydialog
    dd = tmp_path / "dd.parquet"
    pd.DataFrame({"utterances": [["Hi , Tom !", "Hello . How are you ?", "Fine ."], ["Only one"]]}).to_parquet(dd)
    assert load_dailydialog(path=dd) == [f"{USER} Hi, Tom!\n{BOT} Hello. How are you?\n{USER} Fine."]
    bst = tmp_path / "bst.parquet"
    pd.DataFrame({"previous_utterance": [["I love hiking.", "Me too!"]], "free_messages": [["Where do you go?"]],
                  "guided_messages": [["Mostly the Mournes."]]}).to_parquet(bst)
    assert load_bst(path=bst) == [f"{USER} I love hiking.\n{BOT} Me too!\n{USER} Where do you go?\n{BOT} Mostly the Mournes."]
