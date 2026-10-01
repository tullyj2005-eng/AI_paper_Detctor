"""Train the detector and save it.

Run from the project root:
    python -m training.export_model

Writes:
    models/detector.joblib          -> used by the Gradio / Streamlit demos
    chrome_extension/model.json     -> the same model as plain numbers for JavaScript
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dataset_pipeline.dataset import load_sample, build_feature_table, get_xy

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "detector.joblib"
EXTENSION_JSON = ROOT / "chrome_extension" / "model.json"

# Highest share of human essays we are willing to wrongly flag as AI.
MAX_FALSE_POSITIVE_RATE = 0.05


def make_model() -> Pipeline:
    """Same pipeline as training/train.py."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=0)),
    ])


def load_split():
    """Load the data and make the same 80/20 split by prompt that train.py uses."""
    df = load_sample(ROOT / "data" / "sample_2k.csv")
    table = build_feature_table(df)
    X, y, groups, names = get_xy(table)
    X, y, groups = X.to_numpy(dtype=float), y.to_numpy(), groups.to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return df, X, y, groups, names, train_idx, test_idx


def pick_threshold(y_true, proba, max_fpr=MAX_FALSE_POSITIVE_RATE) -> float:
    """Lowest cut-off that still keeps the false positive rate at or under max_fpr."""
    fpr, _, thresholds = roc_curve(y_true, proba)
    i = np.flatnonzero(fpr <= max_fpr)[-1]
    return float(thresholds[i]) if np.isfinite(thresholds[i]) else 0.5


def export_for_javascript(model: Pipeline, names, threshold, path=EXTENSION_JSON):
    """A logistic regression is just a few lists of numbers - write them as JSON."""
    params = {
        "feature_names": list(names),
        "medians": model.named_steps["impute"].statistics_.tolist(),
        "means": model.named_steps["scale"].mean_.tolist(),
        "scales": model.named_steps["scale"].scale_.tolist(),
        "coefficients": model.named_steps["clf"].coef_[0].tolist(),
        "intercept": float(model.named_steps["clf"].intercept_[0]),
        "threshold": threshold,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2))


if __name__ == "__main__":
    df, X, y, groups, names, train_idx, test_idx = load_split()

    model = make_model().fit(X[train_idx], y[train_idx])
    proba = model.predict_proba(X[test_idx])[:, 1]
    threshold = pick_threshold(y[test_idx], proba)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": names, "threshold": threshold}, MODEL_PATH)
    export_for_javascript(model, names, threshold)

    print(f"threshold for <= {MAX_FALSE_POSITIVE_RATE:.0%} false positives: {threshold:.3f}")
    print(f"saved {MODEL_PATH.relative_to(ROOT)} and {EXTENSION_JSON.relative_to(ROOT)}")
