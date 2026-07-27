import os
import pandas as pd
from collections import Counter

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

input_path = os.path.join(
    BASE_DIR,
    "data",
    "annotated",
    "hackernews_annotations_completed.csv"
)

output_dir = os.path.join(BASE_DIR, "data", "annotated")

final_output_path = os.path.join(
    output_dir,
    "hackernews_annotations_final.csv"
)

review_output_path = os.path.join(
    output_dir,
    "hackernews_annotations_review_required.csv"
)

summary_dir = os.path.join(BASE_DIR, "results", "tables", "section_2")

summary_output_path = os.path.join(
    summary_dir,
    "sentiment_distribution_summary.csv"
)

quality_output_path = os.path.join(
    summary_dir,
    "annotation_quality_summary.csv"
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ANNOTATOR_COLUMNS = [
    "annotator_1",
    "annotator_2",
    "annotator_3",
    "annotator_4"
]

VALID_LABELS = {"positive", "neutral", "negative"}

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def normalize_label(label):
    """
    Normalize labels.
    Empty values become None.
    Valid labels are positive, neutral, negative.
    """

    if pd.isna(label):
        return None

    label = str(label).strip().lower()

    if label == "":
        return None

    if label not in VALID_LABELS:
        return "invalid"

    return label


def majority_vote(row):
    """
    Apply majority voting.

    Rules:
    - At least 3 valid annotations are required.
    - If one label has clear majority, return that label.
    - If there is a tie, return manual_review.
    - If fewer than 3 valid labels, return needs_review.
    - If invalid label exists, return invalid_label.
    """

    labels = []

    for col in ANNOTATOR_COLUMNS:
        label = normalize_label(row[col])

        if label == "invalid":
            return "invalid_label"

        if label is not None:
            labels.append(label)

    if len(labels) < 3:
        return "needs_review"

    counts = Counter(labels)

    max_count = max(counts.values())
    winners = [
        label for label, count in counts.items()
        if count == max_count
    ]

    if len(winners) == 1:
        return winners[0]

    return "manual_review"


def count_valid_annotations(row):
    count = 0

    for col in ANNOTATOR_COLUMNS:
        label = normalize_label(row[col])

        if label in VALID_LABELS:
            count += 1

    return count


def count_invalid_annotations(row):
    count = 0

    for col in ANNOTATOR_COLUMNS:
        label = normalize_label(row[col])

        if label == "invalid":
            count += 1

    return count


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    df = pd.read_csv(input_path)

    print("Loaded completed annotations:", len(df), "records")

    required_columns = [
        "item_id",
        "post_id",
        "topic",
        "created_at",
        "url",
        "text_preview"
    ] + ANNOTATOR_COLUMNS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Normalize annotator columns
    for col in ANNOTATOR_COLUMNS:
        df[col] = df[col].apply(normalize_label)

    # Quality checks
    df["valid_annotation_count"] = df.apply(
        count_valid_annotations,
        axis=1
    )

    df["invalid_annotation_count"] = df.apply(
        count_invalid_annotations,
        axis=1
    )

    # Majority voting
    df["final_sentiment"] = df.apply(
        majority_vote,
        axis=1
    )

    # Records requiring review
    review_df = df[
        df["final_sentiment"].isin(
            ["needs_review", "manual_review", "invalid_label"]
        )
    ].copy()

    # Create folders if needed
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    # Save final annotated dataset
    df.to_csv(
        final_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Save review-required records
    review_df.to_csv(
        review_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Sentiment distribution summary
    valid_final_df = df[
        df["final_sentiment"].isin(VALID_LABELS)
    ].copy()

    sentiment_summary = (
        valid_final_df
        .groupby(["topic", "final_sentiment"])
        .size()
        .reset_index(name="count")
    )

    sentiment_summary.to_csv(
        summary_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Annotation quality summary
    quality_summary = pd.DataFrame({
        "metric": [
            "total_records",
            "records_with_final_label",
            "records_needing_review",
            "needs_review_missing_annotations",
            "manual_review_ties",
            "invalid_label_records"
        ],
        "value": [
            len(df),
            len(valid_final_df),
            len(review_df),
            (df["final_sentiment"] == "needs_review").sum(),
            (df["final_sentiment"] == "manual_review").sum(),
            (df["final_sentiment"] == "invalid_label").sum()
        ]
    })

    quality_summary.to_csv(
        quality_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("Final sentiment distribution:")
    print(df["final_sentiment"].value_counts().to_string())

    print()
    print("Final sentiment distribution by topic:")
    print(
        pd.crosstab(
            df["topic"],
            df["final_sentiment"]
        ).to_string()
    )

    print()
    print("Annotation quality:")
    print(quality_summary.to_string(index=False))

    print()
    print("Files saved:")
    print(final_output_path)
    print(review_output_path)
    print(summary_output_path)
    print(quality_output_path)


if __name__ == "__main__":
    main()