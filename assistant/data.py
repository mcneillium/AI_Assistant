"""ConvAI2 dialogues as role-labelled text, split by dialogue.

ConvAI2 (Dinan et al., 2019) is a set of human-to-bot conversations. Each one
becomes a block of lines like

    User: hi, how are you?
    Assistant: i am good, i just got back from the gym.

so the model learns whose turn it is. The split is made by whole dialogue, so
no conversation appears in both training and evaluation.
"""
import json
import random
import re
from pathlib import Path

USER, BOT = "User:", "Assistant:"
# Pinned to a fixed revision of the Hub's parquet conversion, so the data cannot change under us.
CONVAI2_URL = ("https://huggingface.co/datasets/conv_ai_2/resolve/"
               "refs%2Fconvert%2Fparquet/conv_ai_2/train/0000.parquet")


def format_dialogue(turns):
    """turns: [{'sender_class': 'Human'|'Bot', 'text': str}, ...] -> one text block."""
    lines = []
    for t in turns:
        text = " ".join(str(t.get("text", "")).split())
        if not text:
            continue
        role = USER if t.get("sender_class") == "Human" else BOT
        lines.append(f"{role} {text}")
    return "\n".join(lines)


def load_convai2(path=None, min_turns=2):
    """Return a list of formatted dialogues. `path` is a local copy of the parquet file."""
    import pandas as pd
    df = pd.read_parquet(path or CONVAI2_URL)
    out = []
    for turns in df["dialog"]:
        turns = list(turns)
        if len(turns) >= min_turns:
            block = format_dialogue(turns)
            if block.count("\n") + 1 >= min_turns:
                out.append(block)
    return out


def load_custom(path):
    """Optional extra dialogues: JSON lines of {"turns": [{"role": "user"|"assistant", "text": ...}]}."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            turns = json.loads(line)["turns"]
            out.append(format_dialogue(
                [{"sender_class": "Human" if t["role"] == "user" else "Bot", "text": t["text"]} for t in turns]))
    return out


def split(dialogues, eval_share=0.1, seed=42):
    """Shuffle whole dialogues with a fixed seed and hold out a share for evaluation."""
    items = list(dialogues)
    random.Random(seed).shuffle(items)
    n_eval = max(1, int(len(items) * eval_share))
    return items[n_eval:], items[:n_eval]


# ── More dialogue data (2026) ───────────────────────────────────────────────
# DailyDialog (Li et al., 2017): ~13k everyday two-person conversations, written
# by people. Licence CC BY-NC-SA 4.0, so models trained on it are for
# non-commercial use. Blended Skill Talk (Smith et al., 2020): ~7k conversations
# mixing persona, knowledge and empathy, from Facebook AI's ParlAI.
# Both are pinned to the Hub's parquet conversions and keep their own
# validation splits, which are used as held-out sets.
HUB = "https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}/0.parquet"
DAILYDIALOG = ("roskoN/dailydialog", "full")
BST = ("ParlAI/blended_skill_talk", "default")

_SPACE_BEFORE = re.compile(r"\s+([,.!?;:%)])")
_SPACE_AFTER = re.compile(r"([($])\s+")


def detok(text):
    """DailyDialog is pre-tokenised ("Say , Jim , how ' s it going ?"); undo that."""
    t = " ".join(str(text).split())
    t = _SPACE_BEFORE.sub(r"\1", t)
    t = _SPACE_AFTER.sub(r"\1", t)
    t = re.sub(r"\s*([’'])\s*(s|t|re|ve|ll|d|m)\b", r"\1\2", t)
    t = t.replace(" n't", "n't").replace(" ’ ", "’").replace(" ' ", "'")
    return t


def _alternate(utterances):
    """Two speakers taking turns: first is User, second Assistant."""
    return format_dialogue([{"sender_class": "Human" if i % 2 == 0 else "Bot", "text": u}
                            for i, u in enumerate(utterances)])


def load_dailydialog(split="train", path=None):
    import pandas as pd
    df = pd.read_parquet(path or HUB.format(repo=DAILYDIALOG[0], config=DAILYDIALOG[1], split=split))
    return [_alternate([detok(u) for u in row]) for row in df["utterances"] if len(row) >= 2]


def load_bst(split="train", path=None):
    import pandas as pd
    df = pd.read_parquet(path or HUB.format(repo=BST[0], config=BST[1], split=split))
    out = []
    for prev, free, guided in zip(df["previous_utterance"], df["free_messages"], df["guided_messages"]):
        turns = list(prev)
        for a, b in zip(free, guided):
            turns += [a, b]
        if len(turns) >= 2:
            out.append(_alternate(turns))
    return out
