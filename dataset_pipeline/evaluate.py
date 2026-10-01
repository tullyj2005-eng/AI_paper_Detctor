from features import FEATURES 
from features import split_sentences, count_words, burstiness
from pathlib import Path, Path

def load_documents(folder: Path) -> list[tuple[str, str]]:
    """Load all .txt files in the given folder. Return a list of (filename, text) tuples."""
    documents = []
    for file_path in folder.glob("*.txt"):
        file_name = file_path.name
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append((file_path.name, text))
    return documents

def score(documents, feature_fn) -> list[tuple[str, float | None]]: 
    """Score each document using the given feature function. Return a list of (filename, score) tuples."""
    scores = []
    for file_name, text in documents:
        score_value = feature_fn(text)
        scores.append((file_name, score_value))
    return scores

def auc(ai_values: list[float], human_values: list[float]) -> float:
    """Compute the AUC (Area Under the Curve) for the given AI and human feature values."""
    # Sort the values
    ai_values_sorted = sorted(ai_values)
    human_values_sorted = sorted(human_values)

    # Count the number of pairs where AI value is greater than human value
    count = 0
    total_pairs = len(ai_values) * len(human_values)

    for ai in ai_values_sorted:
        for human in human_values_sorted:
            if ai > human:
                count += 1

    return count / total_pairs if total_pairs > 0 else 0.0

if __name__ == "__main__":
    # Load documents
    human = load_documents(Path("data/human"))
    ai = load_documents(Path("data/ai"))
    for name, fn in FEATURES.items():
        print(f"Scoring feature: {name}")
        human_scores = score(human, fn)
        ai_scores = score(ai, fn)
        # Extract just the score values for AUC calculation
        human_values = [score for _, score in human_scores if score is not None]
        ai_values = [score for _, score in ai_scores if score is not None]
        auc_value = auc(ai_values, human_values)
        print(f"AUC for {name}: {auc_value:.4f}")




    