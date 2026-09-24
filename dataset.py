import pandas as pd

essays = pd.read_csv("data/AIDE_train_essays.csv")
prompts = pd.read_csv("data/train_prompts.csv")

print("essays:", essays.shape)
print(essays.dtypes)
print("\nprompts:", prompts.shape)
print(prompts.columns.tolist())
print("\nfirst essay row:")
print(essays.iloc[0])
print(essays["generated"].value_counts())
print(essays.groupby("prompt_id").size())
print(prompts[["prompt_id", "prompt_name"]])
print("\nwords per essay:")
print(essays["text"].str.split().str.len().describe().round(1))

df = pd.read_csv("data/train_v2_drcat_02.csv")   # check the real filename
print(df.shape)
print(df.dtypes)
print(df.iloc[0])
print(df["label"].value_counts())                # or whatever the label column is
print(df.groupby("label")["text"].apply(lambda s: s.str.split().str.len().describe()))
print("duplicate texts:", df["text"].duplicated().sum())

print(df.groupby("label")["text"].apply(lambda s: (s.str.split().str.len() < 150).sum()))

print("-------------------------------------------")
n_words = df["text"].str.split().str.len()
band = df[(n_words >= 250) & (n_words <= 700)].copy()

print(band["label"].value_counts())
print(band.groupby("label")["text"].apply(lambda s: s.str.split().str.len().describe()).round(1))
print(band["source"].value_counts())
print(pd.crosstab(band["prompt_name"], band["label"]))


# data cleaning filters
def balance(g):
    n = g["label"].value_counts().min()
    return g.groupby("label", group_keys=False).sample(n=n, random_state=0)

balanced = band.groupby("prompt_name", group_keys=False).apply(balance)

def word_count(text): return float(len(text.split()))

print(sample["label"].value_counts())
print(pd.crosstab(sample["prompt_name"], sample["label"]))
print(sample.groupby("label")["text"].apply(lambda s: s.str.split().str.len().mean()).round(1))