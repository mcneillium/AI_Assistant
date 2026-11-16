import pytest

TINY = "sshleifer/tiny-gpt2"  # a 2 MB random GPT-2 with the real tokenizer: fast, not meaningful


@pytest.fixture(scope="session")
def tiny():
    from assistant import model as m
    return m.load(TINY)
