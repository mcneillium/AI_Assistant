"""Fine-tune GPT-2 on ConvAI2 and compare it with the untuned model on held-out dialogues.

    python -m assistant.train --out models/gpt2-convai2            # full run
    python -m assistant.train --max-steps 20 --out /tmp/smoke        # quick check

Writes <out>/ (model and tokenizer) and <out>/metrics.json with perplexity on
the held-out dialogues for the base model and the fine-tuned one. Perplexity is
exp(mean token loss) on the same packed 128-token blocks for both, so the two
numbers are directly comparable. Lower is better.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, set_seed)

from .data import CONVAI2_URL, load_convai2, load_custom, split


class Blocks(torch.utils.data.Dataset):
    """All dialogues joined with an end-of-text token, cut into equal blocks."""

    def __init__(self, texts, tok, block):
        ids = []
        for t in texts:
            ids += tok(t)["input_ids"] + [tok.eos_token_id]
        n = len(ids) // block
        self.blocks = [ids[i * block:(i + 1) * block] for i in range(n)]
        self.tokens = n * block

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        return {"input_ids": self.blocks[i]}


def perplexity(trainer, dataset):
    return math.exp(trainer.evaluate(eval_dataset=dataset)["eval_loss"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gpt2")
    ap.add_argument("--data", help="local copy of the ConvAI2 parquet file (default: download)")
    ap.add_argument("--custom", help="optional JSON-lines file of extra dialogues")
    ap.add_argument("--out", default="models/gpt2-convai2")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-share", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    set_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dialogues = load_convai2(a.data)
    train_d, eval_d = split(dialogues, a.eval_share, a.seed)
    if a.custom:
        train_d += load_custom(a.custom)  # extra data only ever goes into training

    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.eos_token
    train_ds, eval_ds = Blocks(train_d, tok, a.block), Blocks(eval_d, tok, a.block)
    model = AutoModelForCausalLM.from_pretrained(a.base)
    args = TrainingArguments(output_dir=str(out / "checkpoints"), num_train_epochs=a.epochs, max_steps=a.max_steps,
                             per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
                             learning_rate=a.lr, warmup_ratio=0.05, weight_decay=0.01, logging_steps=25,
                             save_strategy="no", report_to=[], seed=a.seed, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))

    started = time.time()
    base_ppl = perplexity(trainer, eval_ds)
    print(f"base {a.base}: held-out perplexity {base_ppl:.2f}")
    result = trainer.train()
    tuned_ppl = perplexity(trainer, eval_ds)
    print(f"fine-tuned: held-out perplexity {tuned_ppl:.2f}")
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))

    losses = [(h["step"], round(h["loss"], 4)) for h in trainer.state.log_history if "loss" in h]
    metrics = {
        "base_model": a.base, "data": a.data or CONVAI2_URL,
        "dialogues": {"train": len(train_d), "held_out": len(eval_d)},
        "tokens": {"train": train_ds.tokens, "held_out": eval_ds.tokens}, "block": a.block,
        "steps": result.global_step, "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "seed": a.seed,
        "held_out_perplexity": {"base": round(base_ppl, 2), "fine_tuned": round(tuned_ppl, 2)},
        "train_loss_log": losses, "seconds": round(time.time() - started),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["held_out_perplexity"]))


if __name__ == "__main__":
    main()
