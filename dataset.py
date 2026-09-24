import pandas as pd

from features import FEATURES


def load_sample(path="data/sample_2k.csv") -> pd.DataFrame:
    """Load the sample CSV file and return a DataFrame."""
    return pd.read_csv(path)

def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df[["label", "prompt_name", "source"]].copy()
    for name in sorted(FEATURES):
        table[name] = df["text"].apply(FEATURES[name])
    print(table[sorted(FEATURES)].isna().sum())
    return table

def get_xy(table):
    names = sorted(FEATURES)
    X = table[names]
    y = table["label"]
    groups = table["prompt_name"]
    return X, y, groups, names

if __name__ == "__main__":
    df = load_sample()
    table = build_feature_table(df)
    X, y, groups, names = get_xy(table)
    print(f"\nX: {X.shape}   y: {len(y)}   groups: {len(set(groups))} prompts")
    print(f"features: {names}")

print(table["proper_punctuation_usage"].head())
print(table.groupby("label")["proper_punctuation_usage"].describe().round(4))