import os
import pandas as pd

# find project root regardless of where the script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "guardian_posts_raw.csv"))

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