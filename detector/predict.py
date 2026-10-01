## Load the saved model and score one piece of text.

##  from detector.predict import Detector
##  result = Detector().predict("some essay text ...")

from pathlib import Path

import joblib
import numpy as np

from features.features import FEATURES, count_words

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "detector.joblib"
MIN_WORDS = 100  # comma_rate and semicolon_rate need at least this many words


class Detector:
    def __init__(self, path=MODEL_PATH):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.names = bundle["feature_names"]
        self.threshold = bundle["threshold"]

    def features(self, text: str) -> dict:
        return {name: FEATURES[name](text) for name in self.names}

    def predict(self, text: str) -> dict:
        feats = self.features(text)
        x = np.array([[np.nan if feats[n] is None else feats[n] for n in self.names]])
        prob_ai = float(self.model.predict_proba(x)[0, 1])

        # How much each feature pushed the score toward AI (+) or human (-).
        z = self.model.named_steps["scale"].transform(
            self.model.named_steps["impute"].transform(x))[0]
        coefs = self.model.named_steps["clf"].coef_[0]
        contributions = dict(zip(self.names, (z * coefs).tolist()))

        return {
            "prob_ai": prob_ai,
            "is_ai": prob_ai >= self.threshold,
            "threshold": self.threshold,
            "word_count": count_words(text),
            "too_short": count_words(text) < MIN_WORDS,
            "features": feats,
            "contributions": contributions,
        }


if __name__ == "__main__":
    import sys
    text = sys.stdin.read() if not sys.stdin.isatty() else input("Paste text: ")
    r = Detector().predict(text)
    print(f"P(AI) = {r['prob_ai']:.1%}  ->  {'AI' if r['is_ai'] else 'human'}")
