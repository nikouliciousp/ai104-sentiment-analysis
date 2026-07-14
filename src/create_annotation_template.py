import os
import pandas as pd
import csv

# find project root regardless of where the script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(BASE_DIR, "data", "clean", "guardian_posts_clean.csv")
output_dir = os.path.join(BASE_DIR, "data", "annotated")
output_path = os.path.join(output_dir, "annotation_template.csv")

# load cleaned dataset
df = pd.read_csv(input_path)

print("Loaded cleaned dataset:", len(df), "articles")

# create readable text preview for human annotation
# The original text is easier for humans to understand than text_clean.
df["text_preview"] = (
    df["text"]
    .astype(str)
    .str.replace(r"[\r\n\t]+", " ", regex=True)
    .str.replace("\u2028", " ", regex=False)
    .str.replace("\u2029", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    .str.slice(0, 1500)
)

# keep all articles for annotation
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


# add empty columns for annotators
# allowed values: positive / neutral / negative
annotation["annotator_1"] = ""
annotation["annotator_2"] = ""
annotation["annotator_3"] = ""
annotation["annotator_4"] = ""

# final label after majority voting
annotation["final_sentiment"] = ""

print()
print("Annotation template created:", len(annotation), "articles")

print()
print("Articles per topic:")
print(annotation.groupby("topic").size().to_string())

# create output folder if it does not exist
os.makedirs(output_dir, exist_ok=True)

# save as CSV to upload to Google Sheets
annotation.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
    lineterminator="\n",
    quoting=csv.QUOTE_MINIMAL
)

print()
print("Saved:", output_path)

print()
print("Instructions:")
print("Each annotator fills only their own column independently.")
print("Allowed values: positive / neutral / negative")
print("Use text_preview for annotation and URL if more context is needed.")