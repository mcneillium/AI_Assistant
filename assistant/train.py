"""Fine-tune GPT-2 for conversation and compare it with the untuned model on held-out dialogues.

    python -m assistant.train --sources convai2,dailydialog,bst --out models/gpt2-chat
    python -m assistant.train --sources convai2 --max-steps 20 --out /tmp/smoke      # quick check

Held-out sets: a seeded 10% of ConvAI2 dialogues (it has no public validation
split), and the official validation splits of DailyDialog and Blended Skill
Talk. Perplexity is exp(mean token loss) on packed 128-token blocks (the first
--final-blocks of each held-out set), the same blocks for every model, so the
numbers are comparable. Lower is better.

Checkpoints are written every --save-steps; re-running the same command
resumes from the latest one. The best checkpoint by held-out loss is kept.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

from . import data as D


class Blocks(torch.utils.data.Dataset):
    """Dialogues joined with an end-of-text token, cut into equal blocks."""

    def __init__(self, texts, tok, block, limit=None):
        ids = []
        for t in texts:
            ids += tok(t)["input_ids"] + [tok.eos_token_id]
        n = len(ids) // block
        self.blocks = [ids[i * block:(i + 1) * block] for i in range(n)]
        if limit:
            self.blocks = self.blocks[:limit]
        self.tokens = len(self.blocks) * block

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        return {"input_ids": self.blocks[i]}


def load_sources(names, raw, seed, eval_share):
    """{source: (train_dialogues, held_out_dialogues)}"""
    out = {}
    for name in names:
        if name == "convai2":
            path = Path(raw) / "convai2-train.parquet"
            dialogues = D.load_convai2(path if path.exists() else None)
            out[name] = D.split(dialogues, eval_share, seed)
        elif name == "dailydialog":
            p = lambda s: (Path(raw) / f"dd-{s}.parquet") if (Path(raw) / f"dd-{s}.parquet").exists() else None
            out[name] = (D.load_dailydialog("train", p("train")), D.load_dailydialog("validation", p("validation")))
        elif name == "bst":
            p = lambda s: (Path(raw) / f"bst-{s}.parquet") if (Path(raw) / f"bst-{s}.parquet").exists() else None
            out[name] = (D.load_bst("train", p("train")), D.load_bst("validation", p("validation")))
        else:
            raise SystemExit(f"unknown source {name}")
    return out


def perplexity(model, tok, dataset, batch, bf16):
    args = TrainingArguments(output_dir="/tmp/ppl", per_device_eval_batch_size=batch, report_to=[],
                             bf16=bf16, use_cpu=not torch.cuda.is_available())
    t = Trainer(model=model, args=args, data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    return math.exp(t.evaluate(eval_dataset=dataset)["eval_loss"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gpt2")
    ap.add_argument("--sources", default="convai2", help="comma list: convai2, dailydialog, bst")
    ap.add_argument("--raw", default="data/raw", help="folder with local parquet copies (else downloaded)")
    ap.add_argument("--custom", help="optional JSON-lines file of extra dialogues (training only)")
    ap.add_argument("--compare", nargs="*", default=[], help="other model folders to score on the same blocks")
    ap.add_argument("--out", default="models/gpt2-chat")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-share", type=float, default=0.1)
    ap.add_argument("--eval-blocks", type=int, default=160, help="held-out blocks per source checked during training")
    ap.add_argument("--final-blocks", type=int, default=600, help="held-out blocks per source for the final scores (0 = all)")
    ap.add_argument("--save-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-bf16", action="store_true")
    a = ap.parse_args()

    set_seed(a.seed)
    bf16 = not a.no_bf16 and (torch.cuda.is_available() or torch.backends.mkldnn.is_available())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [s.strip() for s in a.sources.split(",") if s.strip()]
    sources = load_sources(names, a.raw, a.seed, a.eval_share)
    train_d = [x for name in names for x in sources[name][0]]
    if a.custom:
        train_d += D.load_custom(a.custom)
    import random
    random.Random(a.seed).shuffle(train_d)

    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.eos_token
    train_ds = Blocks(train_d, tok, a.block)
    held = {name: Blocks(sources[name][1], tok, a.block, a.final_blocks or None) for name in names}
    watch = torch.utils.data.ConcatDataset([Blocks(sources[n][1], tok, a.block, a.eval_blocks) for n in names])

    model = AutoModelForCausalLM.from_pretrained(a.base)
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch, learning_rate=a.lr,
        warmup_ratio=0.03, weight_decay=0.01, lr_scheduler_type="cosine", logging_steps=50,
        eval_strategy="steps", eval_steps=a.save_steps, save_strategy="steps", save_steps=a.save_steps,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, report_to=[], seed=a.seed, bf16=bf16, use_cpu=not torch.cuda.is_available(),
        dataloader_num_workers=0, disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=watch,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

    started = time.time()
    resume = any((out / "checkpoints").glob("checkpoint-*"))
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))

    scores = {}
    for label, path in [("untuned " + a.base, a.base)] + [(Path(c).name, c) for c in a.compare] + [("this run", str(out))]:
        m = AutoModelForCausalLM.from_pretrained(path)
        # Scored in full precision: bfloat16 shifts perplexity by about 1%, which would blur comparisons between runs.
        scores[label] = {n: round(perplexity(m, tok, held[n], a.batch, False), 2) for n in names}
        print(label, scores[label], flush=True)

    log = trainer.state.log_history
    metrics = {
        "base_model": a.base, "sources": names, "custom": bool(a.custom),
        "dialogues": {n: {"train": len(sources[n][0]), "held_out": len(sources[n][1])} for n in names},
        "tokens": {"train": train_ds.tokens, "held_out": {n: held[n].tokens for n in names}}, "block": a.block,
        "steps": result.global_step, "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "seed": a.seed,
        "bf16": bf16, "best_checkpoint": trainer.state.best_model_checkpoint,
        "held_out_perplexity": scores,
        "train_loss_log": [(h["step"], round(h["loss"], 4)) for h in log if "loss" in h],
        "eval_loss_log": [(h["step"], round(h["eval_loss"], 4)) for h in log if "eval_loss" in h],
        "seconds_this_session": round(time.time() - started),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
