"""Feature distributions (human vs AI) for the progress report.  Run from repo root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import load_sample, build_feature_table

HUMAN, AI = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

PANELS = [
    ("comma_rate", "Commas / 1k words", (0, 160)),
    ("burstiness", "Burstiness", (0, 1.0)),
    ("avg_sentence_length", "Words / sentence", (5, 50)),
    ("word_count", "Essay length (words)", (250, 700)),
]

df = load_sample()
table = build_feature_table(df)
table["word_count"] = df["text"].str.split().str.len()

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9))
for ax, (col, title, (lo, hi)) in zip(axes, PANELS):
    bins = np.linspace(lo, hi, 31)
    for label, color, name in [(0, HUMAN, "Human"), (1, AI, "AI")]:
        vals = table.loc[table.label == label, col].dropna().clip(lo, hi)
        ax.hist(vals, bins=bins, color=color, alpha=0.55, label=name,
                edgecolor="white", linewidth=0.4)
        ax.axvline(vals.median(), color=color, linewidth=1.5)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
axes[0].legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout(pad=0.4)
out = Path(__file__).parent / "features.png"
fig.savefig(out, dpi=220)
print("wrote", out)
