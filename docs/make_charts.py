"""Charts for the README and the case study, drawn from models/gpt2-chat/metrics.json.

Run: python docs/make_charts.py  (writes docs/perplexity-by-source.png and docs/training-loss.png)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
m = json.loads((ROOT / "models/gpt2-chat/metrics.json").read_text())
NAMES = {"convai2": "ConvAI2", "dailydialog": "DailyDialog", "bst": "Blended Skill Talk"}
LABELS = {"untuned gpt2": "GPT-2, not fine-tuned", "gpt2-convai2": "Fine-tuned on ConvAI2 only", "this run": "Fine-tuned on all three"}
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3 (light surface)
INK, MUTED, GRID = "#17171a", "#6d6c67", "#e2e0da"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})

# Perplexity on held-out conversations, per source and model (lower is better)
scores = m["held_out_perplexity"]
models = list(scores)
sources = m["sources"]
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
w = 0.8 / len(models)
for i, label in enumerate(models):
    xs = [j + (i - (len(models) - 1) / 2) * w for j in range(len(sources))]
    vals = [scores[label][s] for s in sources]
    bars = ax.bar(xs, vals, w - 0.03, color=COLOURS[i % 3], label=LABELS.get(label, label), zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.6, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(range(len(sources)), [NAMES.get(s, s) for s in sources])
ax.set_ylabel("Perplexity on held-out conversations (lower is better)")
ax.grid(axis="y", color=GRID, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0), fontsize=10)
ax.set_ylim(0, max(v for sc in scores.values() for v in sc.values()) * 1.22)
ax.set_title("Each model scored on conversations it never trained on", loc="left", fontsize=13, color=INK, pad=12)
fig.tight_layout()
fig.savefig(ROOT / "docs/perplexity-by-source.png")

# Training and validation loss
fig, ax = plt.subplots(figsize=(9, 4.0), dpi=160)
tl = m["train_loss_log"]
el = m["eval_loss_log"]
ax.plot([s for s, _ in tl], [v for _, v in tl], color=COLOURS[0], lw=2, label="Training loss")
ax.plot([s for s, _ in el], [v for _, v in el], color=COLOURS[1], lw=2, marker="o", ms=5, label="Validation loss (held out)")
ax.set_xlabel("Training step (batch of 8 × 128 tokens)")
ax.set_ylabel("Loss")
ax.grid(color=GRID)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
ax.set_title("One epoch over three dialogue datasets, on CPU", loc="left", fontsize=13, color=INK, pad=12)
fig.tight_layout()
fig.savefig(ROOT / "docs/training-loss.png")
print("wrote docs/perplexity-by-source.png and docs/training-loss.png")
