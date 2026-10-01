"""

Run from the project root:
    python -m training.plots
"""
import matplotlib
matplotlib.use("Agg")  # save to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from training.export_model import (
    ROOT, MAX_FALSE_POSITIVE_RATE, load_split, make_model, pick_threshold,
)

FIGURES = ROOT / "figures"
HUMAN, AI = "#2a78d6", "#eb6834"   # blue = human, orange = AI in every graph
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print("saved figures/" + name)


def plot_roc(y_te, proba, threshold):
    fpr, tpr, thr = roc_curve(y_te, proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, label="Random guessing")
    ax.plot(fpr, tpr, lw=2, color=AI, label=f"Model (AUC = {roc_auc_score(y_te, proba):.3f})")
    i = np.argmin(np.abs(thr - threshold))
    ax.plot(fpr[i], tpr[i], "o", ms=8, color=AI, mec="white", mew=2)
    ax.annotate(f"threshold {threshold:.2f}\n{fpr[i]:.0%} false positives, {tpr[i]:.0%} of AI caught",
                (fpr[i], tpr[i]), xytext=(15, -30), textcoords="offset points", color=INK, fontsize=9)
    ax.set(xlabel="False positive rate (human essays flagged as AI)",
           ylabel="True positive rate (AI essays caught)", xlim=(0, 1), ylim=(0, 1.01),
           title="ROC curve on held-out prompts")
    ax.legend(loc="lower right")
    save(fig, "roc_curve.png")


def plot_confusion(y_te, proba, threshold):
    cm = confusion_matrix(y_te, (proba >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (r, c), n in np.ndenumerate(cm):
        ax.text(c, r, n, ha="center", va="center", fontsize=14,
                color="white" if n > cm.max() / 2 else INK)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Human", "AI"], yticklabels=["Human", "AI"],
           xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (threshold {threshold:.2f})")
    save(fig, "confusion_matrix.png")


def plot_coefficients(model, names):
    coefs = model.named_steps["clf"].coef_[0]
    order = np.argsort(coefs)
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(names) + 1))
    ax.barh(np.array(names)[order], coefs[order], height=0.6,
            color=[AI if c > 0 else HUMAN for c in coefs[order]])
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Coefficient (standardized)   <- more human  |  more AI ->",
           title="What the model looks at")
    save(fig, "feature_importance.png")


def plot_feature_distributions(X, y, names):
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    for ax, j in zip(axes.flat, range(len(names))):
        col = X[:, j]
        lo, hi = np.nanpercentile(col, [1, 99])
        bins = np.linspace(lo, hi, 30)
        for label, color, text in [(0, HUMAN, "Human"), (1, AI, "AI")]:
            ax.hist(col[(y == label) & ~np.isnan(col)], bins=bins, alpha=0.55,
                    color=color, label=text)
        ax.set_title(names[j], fontsize=10)
        ax.set_yticks([])
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    axes.flat[0].legend()
    fig.suptitle("How each feature differs between human and AI essays",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "feature_distributions.png")


def plot_probability_histogram(y_te, proba, threshold):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(0, 1, 26)
    ax.hist(proba[y_te == 0], bins=bins, alpha=0.6, color=HUMAN, label="Human essays")
    ax.hist(proba[y_te == 1], bins=bins, alpha=0.6, color=AI, label="AI essays")
    ax.axvline(threshold, color=INK, lw=1.5, ls="--")
    ax.text(threshold, ax.get_ylim()[1] * 0.95, f" threshold {threshold:.2f}", color=INK, va="top")
    ax.set(xlabel="Predicted probability of AI", ylabel="Essays",
           title="Model scores on held-out essays")
    ax.legend(loc="upper center")
    save(fig, "probability_histogram.png")


def plot_cv_vs_baseline(df, X, y, groups, train_idx):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    wc = df["text"].str.split().str.len().to_numpy().reshape(-1, 1)
    results = {
        "Word count only": cross_val_score(make_model(), wc[train_idx], y[train_idx],
                                           groups=groups[train_idx], cv=cv, scoring="roc_auc"),
        "All features": cross_val_score(make_model(), X[train_idx], y[train_idx],
                                        groups=groups[train_idx], cv=cv, scoring="roc_auc"),
    }
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = list(results)
    means = [results[k].mean() for k in labels]
    stds = [results[k].std() for k in labels]
    ax.bar(labels, means, yerr=stds, width=0.5, color=[MUTED, AI], capsize=6,
           error_kw={"ecolor": INK, "lw": 1})
    for i, m in enumerate(means):
        ax.text(i, m + stds[i] + 0.01, f"{m:.3f}", ha="center", color=INK)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set(ylim=(0, 1.05), ylabel="ROC AUC (5-fold CV)", title="Features vs. a word-count baseline")
    ax.grid(axis="x", visible=False)
    save(fig, "cv_vs_baseline.png")


if __name__ == "__main__":
    df, X, y, groups, names, train_idx, test_idx = load_split()
    model = make_model().fit(X[train_idx], y[train_idx])
    proba = model.predict_proba(X[test_idx])[:, 1]
    threshold = pick_threshold(y[test_idx], proba, MAX_FALSE_POSITIVE_RATE)

    plot_roc(y[test_idx], proba, threshold)
    plot_confusion(y[test_idx], proba, threshold)
    plot_coefficients(model, names)
    plot_feature_distributions(X, y, names)
    plot_probability_histogram(y[test_idx], proba, threshold)
    plot_cv_vs_baseline(df, X, y, groups, train_idx)
