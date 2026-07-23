import os
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

clean_input_path = os.path.join(
    BASE_DIR,
    "data",
    "clean",
    "hackernews_posts_clean.csv"
)

labels_input_path = os.path.join(
    BASE_DIR,
    "data",
    "annotated",
    "hackernews_annotations_resolved_final.csv"
)

output_dir = os.path.join(BASE_DIR, "data", "modeling")

modeling_output_path = os.path.join(
    output_dir,
    "hackernews_modeling_dataset.csv"
)

summary_dir = os.path.join(BASE_DIR, "results", "tables")

modeling_quality_output_path = os.path.join(
    summary_dir,
    "modeling_dataset_quality_summary.csv"
)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    clean_df = pd.read_csv(clean_input_path)
    labels_df = pd.read_csv(labels_input_path)

    print("Loaded clean dataset:", len(clean_df), "records")
    print("Loaded resolved labels dataset:", len(labels_df), "records")

    required_clean_columns = [
        "item_id",
        "post_id",
        "text_raw",
        "text_clean",
        "created_at",
        "url",
        "topic",
        "word_count"
    ]

    required_label_columns = [
        "post_id",
        "final_sentiment"
    ]

    missing_clean = [
        col for col in required_clean_columns
        if col not in clean_df.columns
    ]

    missing_labels = [
        col for col in required_label_columns
        if col not in labels_df.columns
    ]

    if missing_clean:
        raise ValueError(f"Missing columns in clean dataset: {missing_clean}")

    if missing_labels:
        raise ValueError(f"Missing columns in labels dataset: {missing_labels}")

    # Keep only one final label per post_id
    labels_df = labels_df[
        [
            "post_id",
            "final_sentiment"
        ]
    ].copy()

    # Merge clean text with final labels
    modeling_df = clean_df.merge(
        labels_df,
        on="post_id",
        how="inner"
    )

    valid_labels = ["negative", "neutral", "positive"]

    modeling_df = modeling_df[
        modeling_df["final_sentiment"].isin(valid_labels)
    ].copy()

    # Final quality checks
    total_records = len(modeling_df)
    missing_text_clean = modeling_df["text_clean"].isna().sum()
    missing_final_sentiment = modeling_df["final_sentiment"].isna().sum()
    duplicate_post_id = modeling_df["post_id"].duplicated().sum()
    min_word_count = modeling_df["word_count"].min()
    max_word_count = modeling_df["word_count"].max()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    modeling_df.to_csv(
        modeling_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    quality_summary = pd.DataFrame({
        "metric": [
            "total_records",
            "missing_text_clean",
            "missing_final_sentiment",
            "duplicate_post_id",
            "min_word_count",
            "max_word_count",
            "negative_records",
            "neutral_records",
            "positive_records"
        ],
        "value": [
            total_records,
            missing_text_clean,
            missing_final_sentiment,
            duplicate_post_id,
            min_word_count,
            max_word_count,
            (modeling_df["final_sentiment"] == "negative").sum(),
            (modeling_df["final_sentiment"] == "neutral").sum(),
            (modeling_df["final_sentiment"] == "positive").sum()
        ]
    })

    quality_summary.to_csv(
        modeling_quality_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("Modeling dataset created:", len(modeling_df), "records")

    print()
    print("Topic distribution:")
    print(modeling_df["topic"].value_counts().to_string())

    print()
    print("Sentiment distribution:")
    print(modeling_df["final_sentiment"].value_counts().to_string())

    print()
    print("Word count range:")
    print("Min:", min_word_count)
    print("Max:", max_word_count)

    print()
    print("Quality summary:")
    print(quality_summary.to_string(index=False))

    print()
    print("Files saved:")
    print(modeling_output_path)
    print(modeling_quality_output_path)


if __name__ == "__main__":
    main()