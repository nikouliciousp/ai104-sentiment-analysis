import pandas as pd

df = pd.read_csv("data/raw/guardian_posts_raw.csv")

print("Total:", len(df))
print()
print(df.groupby("topic").agg(
    articles  = ("post_id",    "count"),
    date_from = ("created_at", "min"),
    date_to   = ("created_at", "max"),
).to_string())
print()
print("Sample text:")
print(df["text"].iloc[0][:300])
