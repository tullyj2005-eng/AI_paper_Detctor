import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedGroupKFold, GroupShuffleSplit, cross_val_score,
)
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from dataset import load_sample, build_feature_table, get_xy

from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, roc_curve,
)

df = load_sample()
table = build_feature_table(df)
X, y, groups, names = get_xy(table)

X = X.to_numpy()
y = y.to_numpy()
groups = groups.to_numpy()

model = Pipeline([
    ("impute", SimpleImputer(strategy="median")),  ## some features return None on short text, median fills in any features returning NaN
    ("scale", StandardScaler()),                   ## burstiness and comma rate are both ranges, logistic regression would think the scale is important this prevents that
    ("clf", LogisticRegression(max_iter=1000, random_state=0)),
])

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_tr, y_tr, g_tr = X[train_idx], y[train_idx], groups[train_idx]
X_te, y_te = X[test_idx], y[test_idx]

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X_tr, y_tr, groups=g_tr, cv=cv, scoring="roc_auc")
print(scores.mean(), scores.std(), np.round(scores, 3))

model.fit(X_tr, y_tr)
coefs = model.named_steps["clf"].coef_[0]
for name, c in sorted(zip(names, coefs), key=lambda p: -abs(p[1])):
    print(f"{name:26} {c:+.3f}")



word_counts = df["text"].str.split().str.len().to_numpy().reshape(-1, 1)
wc_tr = word_counts[train_idx]

baseline = cross_val_score(model, wc_tr, y_tr, groups=g_tr, cv=cv, scoring="roc_auc")
print(f"word-count baseline: {baseline.mean():.4f} +/- {baseline.std():.4f}")
print(f"full feature set:    {scores.mean():.4f} +/- {scores.std():.4f}")

model.fit(X_tr, y_tr)
proba = model.predict_proba(X_te)[:, 1]
pred = (proba >= 0.5).astype(int)
print("held-out AUC:", roc_auc_score(y_te, proba))
print("accuracy:    ", accuracy_score(y_te, pred))
print(confusion_matrix(y_te, pred, labels=[0, 1]))

# --- cost of the default threshold -------------------------------------
tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
print(f"\nat threshold 0.50:")
print(f"  false positive rate:   {fp / (fp + tn):.1%}"
      f"   ({fp} of {fp + tn} human essays flagged as AI)")
print(f"  recall on AI essays:   {tp / (tp + fn):.1%}")
print(f"  precision on AI calls: {tp / (tp + fp):.1%}"
      f"   (of everything flagged, this share really was AI)")

# --- what the detector can do at low false-positive rates --------------
fpr, tpr, thresholds = roc_curve(y_te, proba)
print("\noperating points:")
for target in [0.01, 0.02, 0.05, 0.10]:
    i = min(int(np.searchsorted(fpr, target)), len(thresholds) - 1)
    print(f"  at {fpr[i]:.1%} false positive rate:"
          f" threshold {thresholds[i]:.3f},"
          f" catches {tpr[i]:.1%} of AI essays")