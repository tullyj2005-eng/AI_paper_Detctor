import pandas as pd

# 1. load
df = pd.read_csv("data/train_V2_drcat_02.csv")

# 2. filter to the overlapping length band
n_words = df["text"].str.split().str.len()
band = df[(n_words >= 250) & (n_words <= 700)].copy()

# 3. balance each prompt, then take 2000
parts = []
for name, g in band.groupby("prompt_name"):
    n = g["label"].value_counts().min()
    parts.append(g.groupby("label", group_keys=False).sample(n=n, random_state=0))
balanced = pd.concat(parts)

sample = balanced.sample(n=2000, random_state=0)
sample.to_csv("data/sample_2k.csv", index=False)

# 4. verify
print(sample["label"].value_counts())
print(pd.crosstab(sample["prompt_name"], sample["label"]))
print(sample.groupby("label")["text"].apply(lambda s: s.str.split().str.len().mean()).round(1))