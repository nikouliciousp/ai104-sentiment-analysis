import os
import pandas as pd
import html
import re

# find project root regardless of where the script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(BASE_DIR, "data", "raw", "hackernews_posts_raw.csv")
output_dir = os.path.join(BASE_DIR, "data", "clean")
output_path = os.path.join(output_dir, "hackernews_posts_clean.csv")


def clean_text(text):
    text = str(text)

    # Convert HTML entities such as &#x27;, &quot;, &#x2F;
    text = html.unescape(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove punctuation and special characters, keep letters, numbers and spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    df = pd.read_csv(input_path)

    print("Loaded:", len(df), "records")

    print()
    print("Initial topic distribution:")
    print(df["topic"].value_counts().to_string())

    # Remove duplicates
    before_duplicates = len(df)

    df = df.drop_duplicates(subset=["post_id"]).copy()
    df = df.drop_duplicates(subset=["text_raw", "topic"]).copy()

    after_duplicates = len(df)

    print()
    print("Removed duplicates:", before_duplicates - after_duplicates)
    print("Remaining after duplicate removal:", after_duplicates)

    # Clean text again to ensure consistency
    df["text_raw"] = df["text_raw"].astype(str).apply(html.unescape)
    df["text_clean"] = df["text_raw"].apply(clean_text)

    # Recalculate word count using readable raw text
    df["word_count"] = df["text_raw"].astype(str).apply(
        lambda x: len(str(x).split())
    )

    # Keep only short texts around 10 words
    before_length_filter = len(df)

    df = df[
        (df["word_count"] > 5) &
        (df["word_count"] <= 12)
    ].copy()

    after_length_filter = len(df)

    print()
    print("Removed records outside word-count range:", before_length_filter - after_length_filter)
    print("Remaining after word-count filtering:", after_length_filter)

    # Reset index and add item_id
    df = df.reset_index(drop=True)

    if "item_id" in df.columns:
        df = df.drop(columns=["item_id"])

    df.insert(0, "item_id", range(1, len(df) + 1))

    # Reorder columns
    df = df[
        [
            "item_id",
            "post_id",
            "text_raw",
            "text_clean",
            "created_at",
            "url",
            "topic",
            "word_count"
        ]
    ]

    print()
    print("Final records per topic:")
    print(df["topic"].value_counts().to_string())

    print()
    print("Word count summary:")
    print(df["word_count"].describe())

    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print()
    print("Saved:", output_path)

    print()
    print("Example text_raw:")
    print(df["text_raw"].iloc[0])

    print()
    print("Example text_clean:")
    print(df["text_clean"].iloc[0])


if __name__ == "__main__":
    main()