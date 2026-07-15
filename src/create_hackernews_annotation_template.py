import os
import html
import pandas as pd
import csv

# find project root regardless of where the script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(BASE_DIR, "data", "clean", "hackernews_posts_clean.csv")
output_dir = os.path.join(BASE_DIR, "data", "annotated")
output_path = os.path.join(output_dir, "hackernews_annotation_template.csv")

df = pd.read_csv(input_path)

print("Loaded cleaned dataset:", len(df), "records")

# Create readable preview for human annotation
df["text_preview"] = (
    df["text"]
    .astype(str)
    .apply(html.unescape)
    .str.replace(r"[\r\n\t]+", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    .str.slice(0, 1000)
)

annotation = df[
    [
        "item_id",
        "post_id",
        "topic",
        "created_at",
        "url",
        "text_preview"
    ]
].copy()

annotation["annotator_1"] = ""
annotation["annotator_2"] = ""
annotation["annotator_3"] = ""
annotation["annotator_4"] = ""
annotation["final_sentiment"] = ""

os.makedirs(output_dir, exist_ok=True)

annotation.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
    lineterminator="\n",
    quoting=csv.QUOTE_MINIMAL
)

print()
print("Annotation template created:", len(annotation), "records")

print()
print("Records per topic:")
print(annotation["topic"].value_counts().to_string())

print()
print("Saved:", output_path)