"""Loading the model and generating replies.

The 2024 version passed temperature, top_k and top_p to generate() without
do_sample=True, so Transformers ignored them and decoded greedily. Sampling is
now switched on explicitly, the prompt carries role labels, generation is
limited by new tokens (not total length), and the reply stops at the next turn.
"""
import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import BOT, USER


@dataclass
class GenerationSettings:
    max_new_tokens: int = 48
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    no_repeat_ngram_size: int = 3
    seed: int | None = None


def load(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    return model, tok


def build_prompt(message, history=()):
    """history: [(user_text, assistant_text), ...] oldest first."""
    lines = []
    for u, a in history:
        lines += [f"{USER} {u.strip()}", f"{BOT} {a.strip()}"]
    lines += [f"{USER} {message.strip()}", BOT]
    return "\n".join(lines)


# The reply ends where the model starts another turn: a newline, our own labels with
# any punctuation ("User:", "User,", "Assistant."), or an invented one ("Manager:").
_NEXT_TURN = re.compile(r"\n|\b(?:User|Assistant)\b\s*[:,.]?|\b[A-Z][a-z]+:")


def clean_reply(text):
    """Keep only the assistant's turn."""
    text = text.lstrip()
    m = _NEXT_TURN.search(text)
    if m:
        text = text[:m.start()]
    return " ".join(text.split()).rstrip(" ,;:")


def reply(model, tok, message, history=(), settings=GenerationSettings(), attempts=3):
    """A cleaned reply; if a sample is empty after cleaning, try again (up to `attempts`)."""
    text = ""
    for i in range(attempts):
        seed = None if settings.seed is None else settings.seed + i
        text = _sample(model, tok, message, history, GenerationSettings(**{**settings.__dict__, "seed": seed}))
        if text:
            break
    return text


def _sample(model, tok, message, history, settings):
    prompt = build_prompt(message, history)
    enc = tok(prompt, return_tensors="pt")
    if settings.seed is not None:
        torch.manual_seed(settings.seed)
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=True,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
            no_repeat_ngram_size=settings.no_repeat_ngram_size,
            pad_token_id=tok.pad_token_id,
        )
    new_tokens = out[0, enc["input_ids"].shape[1]:]
    return clean_reply(tok.decode(new_tokens, skip_special_tokens=True))
