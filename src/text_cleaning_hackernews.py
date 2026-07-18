import os
import re
import html
import pandas as pd

# find project root regardless of where the script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path  = os.path.join(BASE_DIR, "data", "raw",   "hackernews_posts_raw.csv")
output_dir  = os.path.join(BASE_DIR, "data", "clean")
output_path = os.path.join(output_dir, "hackernews_posts_clean.csv")


def clean_text(text):
    text = str(text)

    # convert html entities such as &gt; and &amp;
    text = html.unescape(text)

    # convert to lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", " ", text)

    # remove punctuation and special characters, keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    df = pd.read_csv(input_path)
    print("Loaded:", len(df), "records")

    # remove duplicate records
    before_duplicates = len(df)
    df = df.drop_duplicates(subset=["post_id"]).copy()
    df = df.drop_duplicates(subset=["text_raw", "topic"]).copy()
    after_duplicates  = len(df)
    print("Removed duplicates:", before_duplicates - after_duplicates)
    print("Remaining:", after_duplicates)

    # apply text cleaning to raw text
    df["text_clean"] = df["text_raw"].apply(clean_text)

    # count words after cleaning
    df["word_count"] = df["text_clean"].fillna("").apply(
        lambda x: len(str(x).split())
    )

    # remove very short texts
    before_short = len(df)
    df = df[df["word_count"] >= 5].reset_index(drop=True)
    after_short  = len(df)
    print("Removed short records:", before_short - after_short)
    print("Remaining:", after_short)

    # add numeric item_id starting from 1
    df.insert(0, "item_id", range(1, len(df) + 1))

    print()
    print("Final records per topic:")
    print(df["topic"].value_counts().to_string())

    # save cleaned file
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print()
    print("Saved:", output_path)

    # show before and after example
    print()
    print("Before cleaning:")
    print(df["text_raw"].iloc[0][:300])
    print()
    print("After cleaning:")
    print(df["text_clean"].iloc[0][:300])


if __name__ == "__main__":
    main()