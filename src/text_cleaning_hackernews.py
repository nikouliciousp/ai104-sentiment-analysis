import os
import re
import html
import pandas as pd

# find project root regardless of where the script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(BASE_DIR, "data", "raw", "hackernews_posts_raw.csv")
output_dir = os.path.join(BASE_DIR, "data", "clean")
output_path = os.path.join(output_dir, "hackernews_posts_clean.csv")


def clean_text(text):
    text = str(text)

    # Convert HTML entities such as &gt; and &amp;
    text = html.unescape(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove punctuation and special characters
    # Keep letters, numbers and spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    df = pd.read_csv(input_path)

    print("Loaded:", len(df), "records")

    # Remove duplicate records
    before_duplicates = len(df)

    df = df.drop_duplicates(subset=["post_id"]).copy()
    df = df.drop_duplicates(subset=["text", "topic"]).copy()

    after_duplicates = len(df)

    print("Removed duplicates:", before_duplicates - after_duplicates)
    print("Remaining after duplicate removal:", after_duplicates)

    # Apply text cleaning
    df["text_clean"] = df["text"].apply(clean_text)

    # Count words after cleaning
    df["word_count"] = df["text_clean"].fillna("").apply(
        lambda x: len(str(x).split())
    )

    # Remove very short texts
    before_short = len(df)
    df = df[df["word_count"] >= 5].reset_index(drop=True)
    after_short = len(df)

    print("Removed short records:", before_short - after_short)
    print("Remaining after short text removal:", after_short)

    # Add numeric item_id starting from 1
    df.insert(0, "item_id", range(1, len(df) + 1))

    print()
    print("Final records per topic:")
    print(df["topic"].value_counts().to_string())

    # Save cleaned file
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print()
    print("Saved:", output_path)

    print()
    print("Example before cleaning:")
    print(df["text"].iloc[0][:300])

    print()
    print("Example after cleaning:")
    print(df["text_clean"].iloc[0][:300])


if __name__ == "__main__":
    main()